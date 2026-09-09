"""Grouping similar per-sequence failures, and the one thing Slow may write.

Grouping
--------
The simplest thing that is actually usable, built out of parts that already
exist rather than a new platform.  There is no embedding store, no clustering,
no new hash.

* ``group_fault.group_first_faults`` supplies the base key -- (full workflow
  fingerprint, response sign) -- over the **per-sequence** Episodes.  Since a
  decision is now one sequence, an Episode's ``per_view_gain`` is that
  sequence's own reading and the group is a set of sequences that ran a
  comparable program and failed in the same direction.
* Comparability is checked before anything is grouped: the Task/Consumer key
  (``task_consumer_key``: task type | model class | metric) and the evaluation
  semantics must be identical across the members.  A group that fails this is
  refused rather than reported.
* ``fault_cases.selectable_fault_types`` supplies the failure symptom, from
  mechanical evidence only.  A fault class with no evidence behind it is
  masked -- that module exists precisely so a failure cannot be attributed by
  assertion -- and a group whose evidence supports nothing lands on
  ``NO_ACTIONABLE_FAULT`` / ``INSUFFICIENT_EVIDENCE``.
* The visible Pattern of each member is carried as the per-sequence public
  feature card, and the group records the spread of each feature across its
  members, so "do these sequences share an observable condition" is a question
  the reader (and Slow) can actually answer.
* ``group_fault.build_contrast_capsule`` supplies the matched **successes**:
  Episodes from the same corpus with the same workflow fingerprint that came
  out POSITIVE or CONFLICT.  Every group carries its contrast cases; a failure
  with too little evidence stays an Episode and is reported as ungrouped
  rather than being pushed into a group to make one.

Nothing here groups by "they both failed" alone and nothing groups by dataset
name.

What Slow may write
-------------------
One thing: **ADD one capability entry whose body is guidance**, on the single
authorised surface ``skill_library.entries/{skill_id}`` under the registered
cause ``SKILL_LIBRARY_GAP``.  Three consequences, all deliberate:

* It is an ADD, so no existing card is overwritten.  DEV-AUTO-3 measured what
  the alternative costs: a PATCH is in-place and single-slot, a failed revision
  does not fall back to the old body, and one revocation took the ancestor
  program down with it (u23, -0.595).  Knowledge that is still working is not
  erased by an update that fails.
* The body may **not** carry ``Frozen program steps:`` and the entry may not
  carry the ``supplies_candidates`` authority.  A capability card with frozen
  steps is candidate supply -- it would make the update "one replacement
  program for the whole group", which is the shape this package exists to stop
  reinstating.  Guidance changes what Fast *proposes for each sequence*; it
  does not hand every member the same answer.  The runtime refuses a manifest
  that violates this, fails closed, and records it.
* It cannot approve itself.  It supplies no candidate, so it carries no
  execution right of any kind -- ``guidance_preflight`` is what enforces that,
  and the manifest is otherwise taken exactly as the model wrote it -- and
  adoption is decided by the deterministic validator below on feedback the
  proposal never saw, not by the model that proposed it.

Validation, registered before the run
-------------------------------------
An update is adopted only if all three lines pass on **withheld** feedback --
a later unit whose readings were not in the Slow input:

L1  the applicability predicate is a *condition*: it matches at least one and
    at most n-1 of the withheld decision population.  A predicate that matches
    everything or nothing states no condition.
L2  the predicate covers the evidence it was raised from: it matches more than
    half of the failing sequences in its own group.
L3  the diagnosis is not contradicted: the group's own programs, re-read on the
    group's own sequences at the withheld window and on the face the group's
    failures were read on, must not have mostly turned materially positive.
    If they had, the failure the guidance explains is not a repeatable one.

L3 costs Consumer fits and no LLM calls.  Failing any line leaves the library
exactly as it was and the package reports that no qualifying update formed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.main_protocol_p4 import per_sequence as ps
from SelfEvolvingHarnessTS.methods.ttha import fault_cases, group_fault
from SelfEvolvingHarnessTS.methods.ttha.retrieval import evaluate_applicability


def applicability_grammar() -> dict[str, Any]:
    """The closed leaf grammar, read out of the contract schema.

    The first live Slow call of this package returned a predicate the schema
    rejected, twice.  The vocabulary is closed and small, so the fix is to
    hand it over rather than to loosen the validator: nothing here widens what
    is legal, it only stops asking the model to guess it.
    """
    import json as _json  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    path = (_Path(__file__).resolve().parents[2] / "contracts" / "schemas"
            / "observable_feature_v1.json")
    numeric: list[str] = []
    boolean: list[str] = []
    bins: list[str] = []
    try:
        doc = _json.loads(path.read_text(encoding="utf-8"))
        for branch in doc.get("oneOf") or ():
            feature = ((branch.get("properties") or {}).get("feature") or {})
            value = ((branch.get("properties") or {}).get("value") or {})
            names = feature.get("enum") or (
                [feature["const"]] if feature.get("const") else [])
            if value.get("type") == "number":
                numeric.extend(str(name) for name in names)
            elif value.get("type") == "boolean":
                boolean.extend(str(name) for name in names)
            elif value.get("enum"):
                names_seen = {str(v) for v in value["enum"]}
                if names_seen <= {"zero", "very_low", "low", "medium",
                                  "high"}:
                    bins.extend(sorted(names_seen))
    except Exception:  # noqa: BLE001 - an unreadable schema is reported empty
        pass
    return {
        "leaf": {"feature": "<name>", "op": "one of >, >=, <, <=, ==",
                 "value": "<number>"},
        "combine": ["{\"all\": [leaf, ...]}", "{\"any\": [leaf, ...]}",
                    "{\"not\": leaf}"],
        "features_taking_a_number": sorted(set(numeric)),
        "features_taking_a_boolean": sorted(set(boolean)),
        "bin_names_some_features_also_accept": sorted(set(bins)),
        "task_kind_is_special": (
            "{\"feature\": \"task_kind\", \"op\": \"==\", \"value\": "
            "\"forecast\"} -- an eligibility gate, not a condition on its own"),
        "a_legal_example": {"all": [
            {"feature": "task_kind", "op": "==", "value": "forecast"},
            {"feature": "local_robust_z_peak", "op": ">=", "value": 3.0}]},
        "anything_else_is_rejected_by_the_schema": True,
    }

MATERIAL = ps.MATERIAL

#: The single writable surface offered to Slow at a batch boundary.
GUIDANCE_SURFACE_TEMPLATE = "skill_library.entries/{skill_id}"
GUIDANCE_CAUSE = "SKILL_LIBRARY_GAP"
FROZEN_PROGRAM_MARKER = "Frozen program steps:"


# ---------------------------------------------------------------------------
# reading an Episode
# ---------------------------------------------------------------------------

def episode_uid(episode: Any) -> str | None:
    uids = (getattr(episode, "context_summary", {}) or {}).get("series_uids") or []
    return str(uids[0]) if uids else None


def episode_steps(episode: Any) -> tuple[tuple[str, dict], ...]:
    geometry = ((getattr(episode, "context_summary", {}) or {})
                .get("program_geometry") or {})
    return ps.normalise_steps(geometry.get("program_steps") or ())


def episode_gains(episode: Any) -> tuple[float | None, float | None]:
    support = (getattr(episode, "support_response", {}) or {}).get("gain")
    delayed_row = getattr(episode, "delayed_response", {}) or {}
    delayed = delayed_row.get("gain") if delayed_row.get("evaluated") else None
    return (float(support) if isinstance(support, (int, float)) else None,
            float(delayed) if isinstance(delayed, (int, float)) else None)


def episode_origin(episode: Any) -> int | None:
    value = (getattr(episode, "context_summary", {}) or {}).get("support_origin")
    return int(value) if isinstance(value, (int, float)) else None


def _symptom(episode: Any) -> str:
    support, delayed = episode_gains(episode)
    if support is None:
        return "UNREADABLE"
    if support < -MATERIAL and delayed is not None and delayed >= MATERIAL:
        return "SUPPORT_NEGATIVE_DELAYED_POSITIVE"
    if support < -MATERIAL:
        return "SUPPORT_NEGATIVE"
    if support >= MATERIAL and delayed is not None and delayed < -MATERIAL:
        return "SUPPORT_POSITIVE_DELAYED_NEGATIVE"
    if support >= MATERIAL:
        return "SUPPORT_POSITIVE"
    return "NEAR_ZERO"


# ---------------------------------------------------------------------------
# comparability
# ---------------------------------------------------------------------------

def comparability(episodes: Sequence[Any]) -> dict[str, Any]:
    """Task, Consumer and evaluation semantics must be one before grouping."""
    keys = sorted({str(getattr(ep, "task_consumer_key", "") or "")
                   for ep in episodes})
    domains = sorted({str(getattr(ep, "domain_namespace", "") or "")
                      for ep in episodes})
    return {
        "task_consumer_keys": keys,
        "domain_namespaces": domains,
        "comparable": len(keys) == 1 and len(domains) == 1,
        "why": ("one Task x Consumer x metric key and one domain namespace, so "
                "a difference between two members is a difference in what was "
                "done, not in what was being asked"),
    }


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def _feature_spread(uids: Sequence[str],
                    features: Mapping[str, Mapping[str, Any]]
                    ) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    names = sorted({name for uid in uids for name in (features.get(uid) or {})})
    for name in names:
        values = [float(features[uid][name]) for uid in uids
                  if isinstance((features.get(uid) or {}).get(name),
                                (int, float))
                  and not isinstance(features[uid][name], bool)]
        if len(values) < 2:
            continue
        values.sort()
        out[name] = {
            "min": round(values[0], 6),
            "median": round(values[len(values) // 2], 6),
            "max": round(values[-1], 6),
            "n": len(values),
        }
    return out


def _enrich_contrast(rows: Sequence[Mapping[str, Any]],
                     by_id: Mapping[str, Any],
                     features: Mapping[str, Mapping[str, Any]]
                     ) -> list[dict[str, Any]]:
    """A contrast case with the context that makes it a contrast.

    ``build_contrast_capsule`` returns matched cases as
    ``{episode_id, provenance, origin, support_gain}``.  That is enough to
    count them and not enough to reason about them: DEV-SEQ-1 run1 handed Slow
    twelve successes as bare gains with no series identity and no visible
    features, alongside a pattern spread computed over the *failures only*, and
    the real Slow abstained with ``insufficient_public_evidence``.  A
    success without its observable context cannot separate anything from a
    failure, so the fields are attached here.  Nothing is added that the
    Episode corpus does not already hold.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        episode = by_id.get(str(row.get("episode_id")))
        if episode is None:
            out.append(dict(row))
            continue
        uid = episode_uid(episode)
        support, delayed = episode_gains(episode)
        steps = episode_steps(episode)
        out.append({
            "episode_id": str(row.get("episode_id")),
            "series_uid": uid,
            "origin": episode_origin(episode),
            "program": ps.program_label(steps),
            "support_gain": (round(support, 6) if support is not None
                             else None),
            "delayed_gain": (round(delayed, 6) if delayed is not None
                             else "UNKNOWN"),
            "symptom": _symptom(episode),
            "public_features": dict(features.get(uid) or {}),
        })
    return out


def _fault_evidence(members: Sequence[Mapping[str, Any]],
                    corpus: Sequence[Any]) -> dict[str, Any]:
    """Mechanical evidence for the five-way fault choice.  No prose input."""
    signatures = {row["workflow"] for row in members}
    positives = [ep for ep in corpus
                 if group_fault._full_workflow_of(ep) in signatures
                 and (episode_gains(ep)[0] or -1.0) >= MATERIAL]
    support_positive = any(isinstance(row["support_gain"], float)
                           and row["support_gain"] >= MATERIAL
                           for row in members)
    delayed_negative = any(isinstance(row["delayed_gain"], float)
                           and row["delayed_gain"] < -MATERIAL
                           for row in members)
    winner_probed = None
    if positives:
        best = max(positives, key=lambda ep: episode_gains(ep)[0] or 0.0)
        steps = episode_steps(best)
        winner_probed = {"op": steps[0][0] if steps else "identity",
                         "gain": round(episode_gains(best)[0] or 0.0, 6),
                         "on_series": episode_uid(best)}
    evidence = {
        "task_contract_conflict": None,
        "diagnosis_contradiction": None,
        "headroom": None,
        "supply_exhausted": False,
        "winner_probed": winner_probed,
        "agent_chosen": (members[0]["program_steps"][0]["op"]
                         if members and members[0]["program_steps"] else None),
        "support_positive": support_positive,
        "delayed_negative": delayed_negative,
    }
    selectable = fault_cases.selectable_fault_types(evidence)
    return {
        "evidence": evidence,
        "selectable_fault_types": selectable,
        "guard_if_none": fault_cases.default_guard(evidence),
        "decision_fields": {
            "task_consumer": (members[0]["task_consumer_key"]
                              if members else None),
            "workflow_sig": members[0]["workflow"] if members else None,
            "response_class": members[0]["relation"] if members else None,
        },
    }


def group_failures(*, episodes: Sequence[Any],
                   features: Mapping[str, Mapping[str, Any]],
                   min_group: int = 2) -> dict[str, Any]:
    """Per-sequence failures into candidate groups, each with its successes."""
    check = comparability(episodes)
    by_id = {str(getattr(ep, "episode_id", "")): ep for ep in episodes}
    failures = group_fault.iter_failure_episodes(episodes)
    raw_groups = group_fault.group_first_faults(episodes, min_group=min_group)
    grouped_ids: set[str] = set()
    groups: list[dict[str, Any]] = []
    for raw in raw_groups:
        members = []
        for ep in raw["episodes"]:
            uid = episode_uid(ep)
            support, delayed = episode_gains(ep)
            steps = episode_steps(ep)
            members.append({
                "episode_id": str(getattr(ep, "episode_id", "?")),
                "series_uid": uid,
                "origin": episode_origin(ep),
                "workflow": group_fault._full_workflow_of(ep),
                "program_label": ps.program_label(steps),
                "program_steps": [{"op": op, "params": dict(params)}
                                  for op, params in steps],
                "support_gain": (round(support, 6) if support is not None
                                 else None),
                "delayed_gain": (round(delayed, 6) if delayed is not None
                                 else "UNKNOWN"),
                "relation": str(getattr(ep, "relation", "") or ""),
                "task_consumer_key": str(
                    getattr(ep, "task_consumer_key", "") or ""),
                "symptom": _symptom(ep),
                "public_features": dict(features.get(uid) or {}),
            })
            grouped_ids.add(str(getattr(ep, "episode_id", "?")))
        uids = [row["series_uid"] for row in members if row["series_uid"]]
        capsule = group_fault.build_contrast_capsule(
            raw, all_episodes=episodes,
            view_keys={row["episode_id"]: [row["series_uid"]]
                       for row in members if row["series_uid"]})
        contrast = {name: _enrich_contrast(rows, by_id, features)
                    for name, rows in (capsule.get("contrast_cases")
                                       or {}).items()}
        groups.append({
            "group_id": "g_%s_%s" % (raw["workflow"], raw["sign"]),
            "workflow": raw["workflow"],
            "sign": raw["sign"],
            "member_count": len(members),
            "distinct_series": sorted(set(uids)),
            "members": members,
            "symptoms": sorted({row["symptom"] for row in members}),
            "parameter_variants": sorted({
                json.dumps(row["program_steps"], sort_keys=True)
                for row in members}),
            "visible_pattern_spread": _feature_spread(uids, features),
            "visible_pattern_spread_of_the_matched_successes": _feature_spread(
                [row["series_uid"] for row
                 in (contrast.get("positive") or [])
                 if row.get("series_uid")], features),
            "contrast_cases": {
                "positive": contrast.get("positive") or [],
                "conflict": contrast.get("conflict") or [],
                "negative_elsewhere": contrast.get("negative") or [],
                "kept_because": ("a group without its matched successes can "
                                 "only say what failed, never what still "
                                 "works"),
            },
            "fault": _fault_evidence(members, episodes),
        })
    ungrouped = [
        {"episode_id": str(getattr(ep, "episode_id", "?")),
         "series_uid": episode_uid(ep),
         "workflow": group_fault._full_workflow_of(ep),
         "support_gain": round(episode_gains(ep)[0] or 0.0, 6),
         "symptom": _symptom(ep),
         "why": "too little evidence to form a group; kept as an Episode"}
        for ep in failures
        if str(getattr(ep, "episode_id", "?")) not in grouped_ids
    ]
    return {
        "comparability": check,
        "episodes_considered": len(episodes),
        "material_failures": len(failures),
        "min_group": int(min_group),
        "groups": sorted(groups, key=lambda row: (-row["member_count"],
                                                  row["group_id"])),
        "group_count": len(groups),
        "ungrouped_failures": ungrouped,
        "grouping_key": ("full typed workflow fingerprint x response sign, "
                         "then comparability, symptom, parameter variants and "
                         "the visible pattern spread as recorded evidence"),
    }


# ---------------------------------------------------------------------------
# the card Slow reads
# ---------------------------------------------------------------------------

def failure_card(group: Mapping[str, Any], *, pattern_id: str,
                 population_features: Mapping[str, Mapping[str, Any]],
                 vocabulary: Sequence[str] = ()) -> dict[str, Any]:
    """One group as a FailurePatternCard.

    Both faces of the construction units are here, because for a decision that
    passed Support and then lost on the delayed face the delayed number *is*
    the failure, and hiding it would hand Slow a card about nothing.  What is
    withheld instead is a whole later unit, which is what the validator reads.
    """
    members = [{
        "series_uid": row["series_uid"],
        "program": row["program_label"],
        "program_steps": row["program_steps"],
        "support_gain": row["support_gain"],
        "delayed_gain": row["delayed_gain"],
        "symptom": row["symptom"],
        "public_features": row["public_features"],
    } for row in group["members"]]
    return {
        "pattern_id": pattern_id,
        "observable_signature": dict(
            group["members"][0]["public_features"] if group["members"] else {}),
        "what_happened": (
            "%d per-sequence decisions ran %s and came out %s on their own "
            "evaluated sequence.  Each decision was taken for one sequence "
            "from that sequence's own visible context; they are grouped here "
            "because the typed workflow and the response sign match, not "
            "because they merely both failed."
            % (group["member_count"], group["workflow"], group["sign"])),
        "failing_decisions": members,
        "matched_successes": group["contrast_cases"]["positive"],
        "matched_conflicts": group["contrast_cases"]["conflict"],
        "visible_pattern_spread": {
            "of_the_failing_decisions": group["visible_pattern_spread"],
            "of_the_matched_successes": group[
                "visible_pattern_spread_of_the_matched_successes"],
            "how_to_read_it": (
                "the same deployment-visible features on both sides.  A "
                "feature whose ranges overlap completely separates nothing; a "
                "feature whose ranges are disjoint is a candidate condition. "
                "Neither is proof -- it is what you have to reason from."),
        },
        "selectable_fault_types": group["fault"]["selectable_fault_types"],
        "fault_guard_if_none": group["fault"]["guard_if_none"],
        "observable_feature_vocabulary": list(vocabulary),
        "applicability_grammar": applicability_grammar(),
        "what_you_may_write": (
            "Exactly one new capability entry whose body is guidance in prose: "
            "what to observe, how to construct candidates, and under which "
            "condition the guidance applies.  It must NOT contain the string "
            "'Frozen program steps:' and must not claim candidate-supply "
            "authority: one replacement program handed to the whole group is "
            "the thing this boundary exists to avoid.  Fast will read the body "
            "and generate its own program for each sequence separately.  State "
            "the condition in observable_applicability over the deployment-"
            "visible features above; a condition that matches every sequence "
            "is not a condition."),
        "how_to_fill_the_manifest": {
            "operation": "ADD",
            "surface_precondition": {"kind": "ABSENT"},
            "new_value.skill_kind": "capability",
            "new_value.skill_id": (
                "a new lowercase id, not one already in "
                "existing_entry_inventory"),
            "new_value.body": "the guidance prose",
            "new_value.allowed_tools": "[]",
            "predicted_agent_behavior_change": (
                "exactly ['retrieve_skill:<your new_value.skill_id>'] -- the "
                "predicate vocabulary is closed and this is the only member "
                "that fits a guidance entry"),
            "observable_applicability": (
                "the condition, written in applicability_grammar's closed leaf "
                "form over the features shown per sequence above; the same "
                "object goes in both edit_manifest.observable_applicability "
                "and new_value.observable_applicability"),
        },
        "you_may_offer_a_provisional_explanation": (
            "A tentative account is acceptable and expected; you are not "
            "required to prove a unique cause first.  Do not force the answer "
            "into 'narrow the Scope' if the evidence does not point there."),
    }


# ---------------------------------------------------------------------------
# proposing, applying, validating
# ---------------------------------------------------------------------------

class GuidanceRefused(RuntimeError):
    """The proposal is not the one thing this boundary authorises."""


def guidance_preflight(manifest: Any) -> None:
    """Fail closed on anything but a guidance ADD."""
    operation = getattr(getattr(manifest, "operation", None), "value",
                        getattr(manifest, "operation", None))
    if str(operation) != "ADD":
        raise GuidanceRefused("only ADD is authorised at this boundary")
    value = dict(getattr(manifest, "new_value", None) or {})
    if str(value.get("skill_kind")) != "capability":
        raise GuidanceRefused("the entry must be a capability entry")
    body = str(value.get("body") or "")
    if FROZEN_PROGRAM_MARKER in body:
        raise GuidanceRefused(
            "a frozen program in the body would make this one replacement "
            "program for the whole group; guidance only")
    guards = dict(value.get("risk_guards") or {})
    authority = dict(guards.get("authority") or {})
    if authority.get("supplies_candidates") is True:
        raise GuidanceRefused("candidate-supply authority is not granted here")
    if value.get("serving_scope"):
        raise GuidanceRefused(
            "the execution boundary is the sequence the decision belongs to; "
            "a serving scope on a guidance card would re-broadcast it")


def build_catalog(*, controller: Any, parent: Any) -> tuple[dict[str, Any], ...]:
    """The single ADD surface, read from the registry rather than written here."""
    definition = next(
        (item for item in controller.surfaces.definitions
         if item.surface_template_id == GUIDANCE_SURFACE_TEMPLATE), None)
    if definition is None:
        return ()
    authorization = controller.router.allowed_targets(GUIDANCE_CAUSE)
    operation = next((op for op in authorization.allowed_operations
                      if op in definition.allowed_operations), None)
    if operation is None:
        return ()
    dependencies = {key: parent.snapshot.dependency_shas[key]
                    for key in definition.required_dependency_keys
                    if key in parent.snapshot.dependency_shas}
    return ({
        "surface_id": GUIDANCE_SURFACE_TEMPLATE,
        "surface_template_id": definition.surface_template_id,
        "target_class": definition.target_class,
        "surface_type": definition.surface_type,
        "operation": operation,
        "allowed_operations": [operation],
        "surface_precondition": {"kind": definition.precondition},
        "required_dependency_keys": list(definition.required_dependency_keys),
        "dependency_precondition_shas": dependencies,
    },)


def _one_proposal(slow_agent: Any, card: Mapping[str, Any],
                  catalog: Sequence[Mapping[str, Any]], snapshot: Any,
                  task_context: Any) -> dict[str, Any]:
    try:
        manifest = slow_agent.propose_edit(
            card, list(catalog), snapshot,
            manifest_preflight=guidance_preflight,
            task_context=task_context)
    except GuidanceRefused as exc:
        return {"proposed": False, "why": "REFUSED_BY_RUNTIME",
                "detail": str(exc)[:300]}
    except Exception as exc:  # noqa: BLE001 - recorded, never hidden
        return {"proposed": False, "why": "SLOW_STAGE_FAULT",
                "detail": "%s: %s" % (type(exc).__name__, str(exc)[:400])}
    if manifest is None:
        return {"proposed": False, "why": "SLOW_ABSTAINED",
                "detail": str(slow_agent.last_no_proposal_reason or "")[:300]}
    value = dict(getattr(manifest, "new_value", None) or {})
    return {
        "proposed": True,
        "manifest": manifest,
        "skill_id": str(value.get("skill_id") or ""),
        "body": str(value.get("body") or ""),
        "risk_guards": dict(value.get("risk_guards") or {}),
        "observable_applicability": dict(
            value.get("observable_applicability") or {}),
        "predicted_agent_behavior_change": list(
            getattr(manifest, "predicted_agent_behavior_change", ()) or ()),
        "predicted_data_effect": list(
            getattr(manifest, "predicted_data_effect", ()) or ()),
        "falsification_condition": list(
            getattr(manifest, "falsification_condition", ()) or ()),
    }


#: A protocol/schema fault is an instrument failure -- the model returned JSON
#: the closed contract rejects -- so one fresh session is allowed to retry it.
#: Nothing else is retried: a Slow that legitimately abstained, and a proposal
#: the runtime refused on its content, both stand.  The retry can only turn an
#: unreadable answer into a readable one; whether the readable one is adopted
#: is still decided by the validator.
RETRYABLE = ("SLOW_STAGE_FAULT",)


def propose_update(*, slow_factory: Any, card: Mapping[str, Any],
                   catalog: Sequence[Mapping[str, Any]], snapshot: Any,
                   task_context: Any = None,
                   attempts: int = 2) -> dict[str, Any]:
    """The real Slow stage.  Returns the manifest, or why there is none."""
    log: list[dict[str, Any]] = []
    outcome: dict[str, Any] = {"proposed": False, "why": "NO_ATTEMPT_MADE"}
    for attempt in range(1, int(attempts) + 1):
        outcome = _one_proposal(slow_factory(), card, catalog, snapshot,
                                task_context)
        log.append({"attempt": attempt,
                    "proposed": bool(outcome.get("proposed")),
                    "why": outcome.get("why"),
                    "detail": outcome.get("detail")})
        if outcome.get("proposed") or outcome.get("why") not in RETRYABLE:
            break
    outcome["attempts"] = log
    outcome["retry_rule"] = (
        "only a protocol/schema fault is retried, and at most once; an "
        "abstention and a content refusal both stand")
    return outcome


def apply_update(*, controller: Any, store: Any, snapshot: Any,
                 manifest: Any) -> dict[str, Any]:
    """Compile the proposal into a candidate snapshot.  Nothing is activated."""
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (  # noqa: PLC0415
        _resolve_apply_manifest,
    )
    try:
        guidance_preflight(manifest)
        resolved = _resolve_apply_manifest(manifest, snapshot)
        receipt = controller.apply_to_fork(store.materialize(snapshot),
                                           resolved,
                                           confirmed_cause=GUIDANCE_CAUSE)
    except Exception as exc:  # noqa: BLE001 - recorded, never relaxed
        return {"applied": False,
                "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
    return {"applied": True,
            "snapshot": receipt.candidate_snapshot.snapshot,
            "runtime_bundle_sha": receipt.candidate_runtime_bundle_sha,
            "parent_runtime_bundle_sha": receipt.parent_runtime_bundle_sha,
            "target_surface_id": receipt.target_surface_id}


def _matches(applicability: Mapping[str, Any],
             features: Mapping[str, Any]) -> bool:
    try:
        matched, _score = evaluate_applicability(applicability, features)
    except Exception:  # noqa: BLE001 - an unparsable predicate matches nothing
        return False
    return bool(matched)


# DEFECT, found 2026-09-08 and fixed in ``dev_seq2_knowledge`` rather than
# here, so DEV-SEQ-1's receipts stay reproducible from the code that wrote
# them.  Two counting faults, both in the direction of understating coverage
# and overstating evidence:
#
#   * ``construction_features`` reaches this function keyed by series UID
#     alone (see ``run_dev_seq1.knowledge_boundary``), so a decision taken at
#     u8 is tested against u7's visible features.  These units are the same
#     course advancing in time, so the same UID really does appear twice.
#   * the L3 loop iterates group *members*, so one real reading cited by two
#     Episodes of the same series and program is counted as two measurements.
#
# ``dev_seq2_knowledge.FeatureBinding`` keys a card by (series_uid, origin)
# and ``MeasurementLedger`` keys a measurement by the prediction cache's own
# key plus the served UID.  DEV-SEQ-1's L2 refusal was recomputed under the
# corrected binding in ``audit_dev_seq2.dev_seq1_l2_recomputed``: the coverage
# changes (1 of 6 series -> 2 of 7 windows) and the refusal still stands, so
# the fix is not a reason to approve that card.
def validate_update(*, applicability: Mapping[str, Any],
                    group: Mapping[str, Any],
                    construction_features: Mapping[str, Mapping[str, Any]],
                    withheld_population: Sequence[str],
                    withheld_features: Mapping[str, Mapping[str, Any]],
                    withheld_reading: Any) -> dict[str, Any]:
    """The three registered lines, on feedback the proposal never saw."""
    withheld_population = [str(uid) for uid in withheld_population]
    matched_withheld = [uid for uid in withheld_population
                        if _matches(applicability, withheld_features.get(uid)
                                    or {})]
    separates = 1 <= len(matched_withheld) <= max(len(withheld_population) - 1,
                                                  0)

    failing_uids = sorted({row["series_uid"] for row in group["members"]
                           if row["series_uid"]})
    covered = [uid for uid in failing_uids
               if _matches(applicability, construction_features.get(uid) or {})]
    covers = bool(failing_uids) and len(covered) * 2 > len(failing_uids)

    delayed_driven = sum(
        1 for row in group["members"]
        if row["symptom"] == "SUPPORT_POSITIVE_DELAYED_NEGATIVE")
    face = ("delayed" if delayed_driven * 2 > len(group["members"])
            else "support")

    rows: list[dict[str, Any]] = []
    for member in group["members"]:
        uid = member["series_uid"]
        if uid is None or uid not in withheld_population:
            continue
        if not _matches(applicability, withheld_features.get(uid) or {}):
            continue
        steps = ps.normalise_steps(member["program_steps"])
        if not steps:
            continue
        try:
            gain = withheld_reading(steps, [uid], face).get(uid)
        except Exception as exc:  # noqa: BLE001 - recorded as unreadable
            rows.append({"series_uid": uid, "program": member["program_label"],
                         "withheld_gain": "UNREADABLE",
                         "why": "%s: %s" % (type(exc).__name__,
                                            str(exc)[:120])})
            continue
        rows.append({"series_uid": uid, "program": member["program_label"],
                     "construction_support_gain": member["support_gain"],
                     "construction_delayed_gain": member["delayed_gain"],
                     "withheld_face": face,
                     "withheld_gain": (round(float(gain), 6)
                                       if gain is not None else "UNKNOWN")})
    readable = [row for row in rows
                if isinstance(row.get("withheld_gain"), float)]
    reversed_rows = [row for row in readable
                     if row["withheld_gain"] >= MATERIAL]
    contradicted = bool(readable) and len(reversed_rows) * 2 > len(readable)

    lines = {
        "L1_the_predicate_is_a_condition": bool(separates),
        "L2_it_covers_the_evidence_it_cites": bool(covers),
        "L3_the_diagnosis_is_not_contradicted": not contradicted,
    }
    return {
        "lines": lines,
        "qualifies": all(lines.values()),
        "failed_lines": [name for name, ok in lines.items() if not ok],
        "L1": {"withheld_population": len(withheld_population),
               "matched": matched_withheld},
        "L2": {"failing_series": failing_uids, "covered": covered},
        "L3": {"face_read": face,
               "why_that_face": (
                   "the face the group's own failures were read on: delayed "
                   "when most members passed Support and lost later, Support "
                   "otherwise"),
               "rows": rows, "readable": len(readable),
               "reversed_materially_positive": len(reversed_rows),
               "unknown": len(rows) - len(readable)},
        "registered_before_the_run": True,
        "why_these_three": (
            "a condition that names everyone is not a condition; guidance that "
            "does not cover the failures it cites is about a different "
            "population; and a failure that does not repeat on a window the "
            "proposal never saw is not a failure worth writing knowledge about"
        ),
    }


__all__ = [
    "GUIDANCE_CAUSE",
    "GUIDANCE_SURFACE_TEMPLATE",
    "GuidanceRefused",
    "apply_update",
    "build_catalog",
    "comparability",
    "episode_gains",
    "episode_origin",
    "episode_steps",
    "episode_uid",
    "failure_card",
    "group_failures",
    "guidance_preflight",
    "propose_update",
    "validate_update",
]
