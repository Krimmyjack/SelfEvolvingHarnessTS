"""DEV-SEQ-2: the two engineering fixes, the wider edit surface, and adoption.

This module continues DEV-SEQ-1 rather than replacing it.  ``per_sequence`` is
reused unchanged -- the per-sequence request binding, the reading composition
and the ``SequenceView`` executor are the parts DEV-SEQ-1 tested and they are
not re-derived here.  What is new is four things.

1.  Window binding (fix)
------------------------
DEV-SEQ-1 keyed the visible-feature card by series UID alone:

    for position in CONSTRUCTION_POSITIONS:
        for uid, card in features[position].items():
            construction_features.setdefault(uid, dict(card))

``setdefault`` means the *first* construction window won.  These course units
are the same course advancing in time, so the same UID appears at u7 and at
u8 -- and every u8 Episode was then described to Slow, and scored by the
validator, with u7's features.  Here a feature card is keyed by
``(series_uid, origin)`` and there is no fallback: a window with no card of
its own reports a miss instead of borrowing another window's.

2.  Measurement de-duplication (fix)
------------------------------------
A group can hold two Episodes of the *same* sequence running the *same*
program (different windows -- or the same window reached twice).  DEV-SEQ-1's
L3 iterated group members, so one real reading could be counted twice, which
inflates the denominator of "did this mostly reverse".  ``MeasurementLedger``
keys a measurement by the cache's own semantic key --
``ReplayPredictionCache.key(unit, origin, config, steps)`` plus the served
UID -- so referencing the same reading again is the same measurement.  No new
hash and no new store: the key is the one the instrument already bills by.
Episodes are **not** deleted; only the measurement count de-duplicates, and a
different window, program, parameter set or training config is a different
key and stays separate.

3.  A wider, still-narrow edit whitelist
----------------------------------------
DEV-SEQ-1 offered Slow exactly one surface (ADD a guidance capability entry).
The failure it then had to explain was not obviously a missing-capability
failure, and a proposal can only be as good as the space it may propose in.
This package offers the four objects the work order authorises, instantiated
against the parent snapshot:

    skill_library.entries/{skill_id}                     ADD   (new guidance)
    skill_library.entries/{id}.body|.observable_...      PATCH (existing
                                                                guidance only)
    bootstrap_skills.entries/{skill_id}.body             PATCH (observation /
                                                                candidate /
                                                                selection
                                                                procedure text)
    candidate_policy.proposal_guidance                   PATCH
    candidate_policy.selection_guidance                  PATCH

Each catalog entry carries its own precondition SHA, its dependency SHAs, the
current value in full (a PATCH replaces the whole value, so the writer has to
see it), and a plain statement of where in Fast that field is actually read.
The confirmed cause is derived *mechanically from the surface the model
chose* -- not decided in advance and not used to veto a surface.  The
five-way fault taxonomy stays on the card as a statistical label and a
factual hint, which is all the evidence supports.

What stays refused, in ``edit_preflight``, fail-closed: any surface outside
the four objects; ``Frozen program steps:`` anywhere in a body or a patch
value (that is one replacement program broadcast to a group, the shape this
line exists to stop); ``supplies_candidates`` authority; a ``serving_scope``;
a non-guidance capability body (the K0 winner card is a frozen program card,
so its body is not a guidance body and is not offered); and any attempt to
edit the model's own approval rules, the risk guards, verification, retrieval
top_k or the candidate slot counts.

4.  Adoption is decided by real Fast, not by predicate coverage
--------------------------------------------------------------
DEV-SEQ-1 adopted or refused on three registered lines about the *predicate*
(is it a condition, does it cover its evidence, does the old program still
fail).  Those lines are kept here as **diagnostics** -- they are cheap and
they say something -- but they no longer gate anything, because "the
condition covers 1 of 6 failing series" is a statement about coverage, not
about whether the guidance helps.  The gate is instead:

    run real Fast under the parent knowledge (A) and under the candidate
    knowledge (B) on the same withheld unit, same population, same budget,
    and read the paired delayed utility.

``adoption_check`` marks a candidate adoptable when the two readings are
complete and comparable, B's mean paired delayed gain beats A's by the
existing MATERIAL threshold, and B passes the existing authoritative gate on
that unit.  No constant moves, no risk denominator moves, and a candidate
that fails the mark is still carried into the follow-up units as an
explicitly *unadopted diagnostic branch*, so the package can tell "the
guidance has no value" apart from "the adoption rule did not pass it".

This is a development-level adoption rule for this package.  It is not a
production deployment permission and nothing here grants a Shared Capability.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.main_protocol_p4 import dev_seq1_knowledge as know1
from evaluation.main_protocol_p4 import per_sequence as ps
from SelfEvolvingHarnessTS.methods.ttha import fault_cases, group_fault

MATERIAL = ps.MATERIAL
FROZEN_PROGRAM_MARKER = know1.FROZEN_PROGRAM_MARKER

episode_uid = know1.episode_uid
episode_steps = know1.episode_steps
episode_gains = know1.episode_gains
episode_origin = know1.episode_origin
comparability = know1.comparability
applicability_grammar = know1.applicability_grammar
_symptom = know1._symptom


# ---------------------------------------------------------------------------
# 1. window binding
# ---------------------------------------------------------------------------

WindowKey = tuple[str, int]


def window_key(uid: Any, origin: Any) -> WindowKey:
    return (str(uid), int(origin))


def bind_window_features(
    by_position: Mapping[int, Mapping[str, Mapping[str, Any]]],
    origins: Mapping[int, int],
) -> dict[WindowKey, dict[str, Any]]:
    """``{position: {uid: card}}`` -> ``{(uid, origin): card}``.

    Every window keeps its own card.  Two windows of one sequence are two
    entries, and neither overwrites the other.
    """
    out: dict[WindowKey, dict[str, Any]] = {}
    for position, cards in by_position.items():
        origin = int(origins[int(position)])
        for uid, card in (cards or {}).items():
            out[window_key(uid, origin)] = dict(card)
    return out


class FeatureBinding:
    """Window-bound feature cards, with misses recorded rather than filled."""

    def __init__(self, cards: Mapping[WindowKey, Mapping[str, Any]]) -> None:
        self._cards = {window_key(uid, origin): dict(card)
                       for (uid, origin), card in cards.items()}
        self.misses: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._cards)

    def uids(self) -> list[str]:
        return sorted({uid for uid, _origin in self._cards})

    def windows(self) -> list[WindowKey]:
        return sorted(self._cards)

    def get(self, uid: Any, origin: Any) -> dict[str, Any]:
        """This window's own card, or ``{}``.  Never another window's."""
        key = window_key(uid, origin)
        card = self._cards.get(key)
        if card is None:
            self.misses.append({"series_uid": str(uid), "origin": int(origin)})
            return {}
        return dict(card)

    def for_episode(self, episode: Any) -> dict[str, Any]:
        uid = episode_uid(episode)
        origin = episode_origin(episode)
        if uid is None or origin is None:
            return {}
        return self.get(uid, origin)

    def spread(self, windows: Sequence[WindowKey]) -> dict[str, dict[str, Any]]:
        """min/median/max of each numeric feature over these windows."""
        rows = [self._cards.get(window_key(*key)) or {} for key in windows]
        names = sorted({name for row in rows for name in row})
        out: dict[str, dict[str, Any]] = {}
        for name in names:
            values = [float(row[name]) for row in rows
                      if isinstance(row.get(name), (int, float))
                      and not isinstance(row.get(name), bool)]
            if len(values) < 2:
                continue
            values.sort()
            out[name] = {"min": round(values[0], 6),
                         "median": round(values[len(values) // 2], 6),
                         "max": round(values[-1], 6),
                         "n": len(values)}
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows": len(self._cards),
            "distinct_series": len(self.uids()),
            "keyed_by": "(series_uid, origin)",
            "misses": self.misses,
            "no_fallback": ("a window with no card of its own reports a miss; "
                            "it never borrows another window's features"),
        }


# ---------------------------------------------------------------------------
# 2. measurement de-duplication
# ---------------------------------------------------------------------------

class MeasurementLedger:
    """Distinct real readings, keyed by the instrument's own semantic key.

    ``cache`` is a ``ReplayPredictionCache``; its ``key`` already distinguishes
    unit, origin, training config and typed program steps, which is exactly
    the set the work order says must not be merged.  The served UID is added
    because one cache entry holds a row per series.  Nothing new is hashed.
    """

    def __init__(self, cache: Any, config_of: Any) -> None:
        self._cache = cache
        self._config_of = config_of
        self._seen: dict[str, dict[str, Any]] = {}
        self.references = 0

    def measurement_key(self, *, unit: str, origin: int,
                        steps: Sequence[tuple], uid: str) -> str:
        base = self._cache.key(str(unit), int(origin),
                               self._config_of(int(origin)),
                               tuple(steps) or ())
        return "%s|%s" % (base, str(uid))

    def note(self, *, unit: str, origin: int, steps: Sequence[tuple],
             uid: str, value: Any, face: str = "") -> bool:
        """Record one reference.  True the first time this reading is seen."""
        key = self.measurement_key(unit=unit, origin=origin, steps=steps,
                                   uid=uid)
        self.references += 1
        if key in self._seen:
            self._seen[key]["referenced"] += 1
            return False
        self._seen[key] = {
            "series_uid": str(uid), "origin": int(origin), "unit": str(unit),
            "face": str(face), "program": ps.program_label_with_params(steps),
            "value": value, "referenced": 1,
        }
        return True

    def distinct(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._seen.values()]

    def to_dict(self) -> dict[str, Any]:
        rows = self.distinct()
        return {
            "references": self.references,
            "distinct_measurements": len(rows),
            "repeated_references": self.references - len(rows),
            "distinct_series": len({row["series_uid"] for row in rows}),
            "distinct_windows": len({(row["series_uid"], row["origin"])
                                     for row in rows}),
            "key": ("ReplayPredictionCache.key(unit, origin, config, steps) "
                    "plus the served series_uid -- the instrument's own key, "
                    "not a new hash"),
            "rule": ("citing one real reading twice is one measurement; "
                     "different window, program, parameters or training "
                     "config is a different key and stays separate; no "
                     "Episode is deleted by this"),
        }


# ---------------------------------------------------------------------------
# 3. grouping, window-bound
# ---------------------------------------------------------------------------

def _member_row(episode: Any, binding: FeatureBinding) -> dict[str, Any]:
    support, delayed = episode_gains(episode)
    steps = episode_steps(episode)
    origin = episode_origin(episode)
    uid = episode_uid(episode)
    return {
        "episode_id": str(getattr(episode, "episode_id", "?")),
        "series_uid": uid,
        "origin": origin,
        "workflow": group_fault._full_workflow_of(episode),
        "program_label": ps.program_label(steps),
        "program_label_with_params": ps.program_label_with_params(steps),
        "program_steps": [{"op": op, "params": dict(params)}
                          for op, params in steps],
        "support_gain": round(support, 6) if support is not None else None,
        "delayed_gain": round(delayed, 6) if delayed is not None else "UNKNOWN",
        "relation": str(getattr(episode, "relation", "") or ""),
        "task_consumer_key": str(getattr(episode, "task_consumer_key", "") or ""),
        "symptom": _symptom(episode),
        "public_features": binding.for_episode(episode),
        "features_are_from": ("this decision's own window, not the first "
                              "window this series appeared in"),
    }


def _contrast_rows(rows: Sequence[Mapping[str, Any]],
                   by_id: Mapping[str, Any],
                   binding: FeatureBinding) -> list[dict[str, Any]]:
    """A matched case with the context that makes it a contrast."""
    out: list[dict[str, Any]] = []
    for row in rows:
        episode = by_id.get(str(row.get("episode_id")))
        if episode is None:
            out.append(dict(row))
            continue
        out.append(_member_row(episode, binding))
    return out


def group_failures(*, episodes: Sequence[Any], binding: FeatureBinding,
                   min_group: int = 2) -> dict[str, Any]:
    """Light organisation of the material.  Not a verdict on a common cause."""
    check = comparability(episodes)
    by_id = {str(getattr(ep, "episode_id", "")): ep for ep in episodes}
    failures = group_fault.iter_failure_episodes(episodes)
    raw_groups = group_fault.group_first_faults(episodes, min_group=min_group)
    grouped_ids: set[str] = set()
    groups: list[dict[str, Any]] = []
    for raw in raw_groups:
        members = [_member_row(ep, binding) for ep in raw["episodes"]]
        grouped_ids.update(row["episode_id"] for row in members)
        capsule = group_fault.build_contrast_capsule(
            raw, all_episodes=episodes,
            view_keys={row["episode_id"]: [row["series_uid"]]
                       for row in members if row["series_uid"]})
        contrast = {name: _contrast_rows(rows, by_id, binding)
                    for name, rows in (capsule.get("contrast_cases") or {}).items()}
        failing_windows = [(row["series_uid"], row["origin"]) for row in members
                           if row["series_uid"] and row["origin"] is not None]
        success_windows = [(row["series_uid"], row["origin"])
                           for row in (contrast.get("positive") or [])
                           if row.get("series_uid") and row.get("origin") is not None]
        groups.append({
            "group_id": "g_%s_%s" % (raw["workflow"], raw["sign"]),
            "workflow": raw["workflow"],
            "sign": raw["sign"],
            "member_count": len(members),
            "distinct_series": sorted({row["series_uid"] for row in members
                                       if row["series_uid"]}),
            "distinct_windows": sorted(set(failing_windows)),
            "members": members,
            "symptoms": sorted({row["symptom"] for row in members}),
            "parameter_variants": sorted({
                json.dumps(row["program_steps"], sort_keys=True)
                for row in members}),
            "visible_pattern_spread": binding.spread(failing_windows),
            "visible_pattern_spread_of_the_matched_successes":
                binding.spread(success_windows),
            "contrast_cases": {
                "positive": contrast.get("positive") or [],
                "conflict": contrast.get("conflict") or [],
                "negative_elsewhere": contrast.get("negative") or [],
                "kept_because": ("a group without its matched successes can "
                                 "only say what failed, never what still "
                                 "works"),
            },
            "fault": know1._fault_evidence(members, episodes),
        })
    ungrouped = [
        {"episode_id": str(getattr(ep, "episode_id", "?")),
         "series_uid": episode_uid(ep), "origin": episode_origin(ep),
         "workflow": group_fault._full_workflow_of(ep),
         "support_gain": round(episode_gains(ep)[0] or 0.0, 6),
         "symptom": _symptom(ep),
         "why": "too little evidence to form a group; kept as an Episode"}
        for ep in failures
        if str(getattr(ep, "episode_id", "?")) not in grouped_ids]
    return {
        "comparability": check,
        "episodes_considered": len(episodes),
        "material_failures": len(failures),
        "min_group": int(min_group),
        "groups": sorted(groups, key=lambda row: (-row["member_count"],
                                                  row["group_id"])),
        "group_count": len(groups),
        "ungrouped_failures": ungrouped,
        "feature_binding": binding.to_dict(),
        "grouping_key": ("full typed workflow fingerprint x response sign, "
                         "under one Task/Consumer key; symptom, parameter "
                         "variants and the window-bound visible spread are "
                         "recorded evidence, not part of the key"),
        "what_a_group_is_not": ("not a verdict on a common cause.  Slow may "
                                "take a sub-group, give a different reading, "
                                "or abstain; it is not required to explain "
                                "the largest group or to cover most failures"),
    }


# ---------------------------------------------------------------------------
# 4. the editable surfaces
# ---------------------------------------------------------------------------

#: surface template -> the registered cause that authorises it.  Derived from
#: ``evaluation/minipipe/feedback/fault_routes.json``; the cause follows the
#: surface the model chose, it does not gate which surfaces are offered.
CAUSE_FOR_TEMPLATE = {
    "skill_library.entries/{skill_id}": "SKILL_LIBRARY_GAP",
    "skill_library.entries/{skill_id}.body": "SKILL_CONTENT_GAP",
    "skill_library.entries/{skill_id}.observable_applicability": "SCOPE_OVERREACH",
    "bootstrap_skills.entries/{skill_id}.body": "OBSERVATION_PROCEDURE_GAP",
    "candidate_policy.proposal_guidance": "PROPOSAL_CONTROL_GAP",
    "candidate_policy.selection_guidance": "SELECTION_MISS",
}

#: What Fast actually does with each field, stated to the writer.  Checked by
#: the smoke against the rendered Fast system prompt, not asserted here.
REACHES_FAST = {
    "skill_library.entries/{skill_id}":
        ("a new capability entry.  Fast resolves capability entries by "
         "observable_applicability and keeps the top 2, so this body reaches "
         "the inspect/propose/select prompt only for sequences whose visible "
         "features match the condition you write"),
    "skill_library.entries/{skill_id}.body":
        ("the body of an existing capability entry, on the same retrieval "
         "path as above"),
    "skill_library.entries/{skill_id}.observable_applicability":
        ("the condition under which an existing capability entry is retrieved "
         "at all"),
    "bootstrap_skills.entries/{skill_id}.body":
        ("a bootstrap procedure.  Bootstrap entries are retrieved always, for "
         "every sequence, with no applicability filter -- so this text reaches "
         "every decision in the batch"),
    "candidate_policy.proposal_guidance":
        ("carried in controls.candidate_policy in the Fast system prompt for "
         "every stage; it is the standing instruction for how candidates are "
         "proposed"),
    "candidate_policy.selection_guidance":
        ("carried in controls.candidate_policy in the Fast system prompt for "
         "every stage; it is the standing instruction for how one candidate "
         "is chosen or identity is kept"),
}

#: Surfaces the model may never touch in this package, listed so the refusal
#: is a stated boundary rather than an accident of what was offered.
FORBIDDEN_TEMPLATES = (
    "instruction.core",
    "retrieval.capability.top_k",
    "candidate_policy.agent_program_slots",
    "candidate_policy.identity_slots",
    "candidate_policy.total_k",
    "verification.rules",
    "verification.rules.scope_risk_guards",
    "skill_library.entries/{skill_id}.risk_guards",
    "memory.entries/{memory_id}",
)


class EditRefused(RuntimeError):
    """The proposal is not inside this boundary's whitelist.

    ``form=True`` marks a refusal about **shape** on a surface the boundary
    does authorise -- the wrong entry kind, a patch value of the wrong type, a
    missing field.  That is a correctable protocol error and the writer gets
    one bounded retry with the concrete message handed back, the same
    allowance a schema fault gets.

    ``form=False`` (the default) marks a refusal about **substance**: a
    surface outside the whitelist, a frozen program, candidate-supply
    authority, a serving scope.  Those stand.  Re-asking until the model says
    something admissible would be pressing it for an answer, which this
    boundary does not do.
    """

    def __init__(self, message: str, *, form: bool = False) -> None:
        super().__init__(message)
        self.form = bool(form)


def _is_guidance_body(body: str) -> bool:
    return FROZEN_PROGRAM_MARKER not in str(body or "")


def build_catalog(*, controller: Any, parent: Any) -> tuple[dict[str, Any], ...]:
    """The four authorised objects, instantiated against this parent snapshot.

    Each PATCH entry carries the live precondition SHA and the whole current
    value, because a PATCH replaces the value and the writer has to see what
    it is replacing.  An object with no instance in this snapshot is not
    offered and the reason is recorded by the caller.
    """
    snapshot = parent.snapshot
    definitions = {item.surface_template_id: item
                   for item in controller.surfaces.definitions}
    entries: list[dict[str, Any]] = []

    def dependencies(definition: Any) -> dict[str, str]:
        return {key: snapshot.dependency_shas[key]
                for key in definition.required_dependency_keys
                if key in snapshot.dependency_shas}

    def add(template: str, surface_id: str, *, operation: str,
            precondition: Mapping[str, Any],
            current_value: Any = None, note: str = "") -> None:
        definition = definitions.get(template)
        if definition is None or operation not in definition.allowed_operations:
            return
        cause = CAUSE_FOR_TEMPLATE[template]
        try:
            authorization = controller.router.allowed_targets(cause)
        except Exception:  # noqa: BLE001 - an unroutable cause offers nothing
            return
        if operation not in authorization.allowed_operations:
            return
        row = {
            "surface_id": surface_id,
            "surface_template_id": template,
            "target_class": definition.target_class,
            "surface_type": definition.surface_type,
            "operation": operation,
            "allowed_operations": [operation],
            "surface_precondition": dict(precondition),
            "required_dependency_keys": list(definition.required_dependency_keys),
            "dependency_precondition_shas": dependencies(definition),
            "reaches_fast_at": REACHES_FAST[template],
        }
        if current_value is not None:
            row["current_value"] = current_value
        if note:
            row["note"] = note
        entries.append(row)

    # (a) ADD one new guidance capability entry.
    add("skill_library.entries/{skill_id}",
        "skill_library.entries/{skill_id}",
        operation="ADD", precondition={"kind": "ABSENT"},
        note=("write your own lowercase skill_id into both the surface id and "
              "new_value.skill_id; it must not already exist"))

    # (b) PATCH an existing *guidance* capability entry.  A capability whose
    #     body is a frozen program card is an execution body, not guidance,
    #     and is not offered on either of its two PATCH surfaces.
    for skill in snapshot.skills:
        if skill.skill_kind.value != "capability":
            continue
        if not _is_guidance_body(skill.body):
            continue
        for template, value in (
                ("skill_library.entries/{skill_id}.body", skill.body),
                ("skill_library.entries/{skill_id}.observable_applicability",
                 json.loads(json.dumps(skill.observable_applicability,
                                       default=dict)))):
            surface_id = template.replace("{skill_id}", skill.skill_id)
            try:
                sha = controller.surface_precondition_sha(parent, surface_id)
            except Exception:  # noqa: BLE001 - not offerable, not fatal
                continue
            add(template, surface_id, operation="PATCH",
                precondition={"kind": "SHA", "sha": sha}, current_value=value)

    # (c) PATCH a bootstrap procedure body.
    for skill in snapshot.skills:
        if skill.skill_kind.value != "bootstrap_procedure":
            continue
        template = "bootstrap_skills.entries/{skill_id}.body"
        surface_id = template.replace("{skill_id}", skill.skill_id)
        try:
            sha = controller.surface_precondition_sha(parent, surface_id)
        except Exception:  # noqa: BLE001
            continue
        add(template, surface_id, operation="PATCH",
            precondition={"kind": "SHA", "sha": sha}, current_value=skill.body)

    # (d) PATCH the two candidate-policy guidance strings.
    policy = dict(snapshot.candidate_policy)
    for template, field in (("candidate_policy.proposal_guidance",
                             "proposal_guidance"),
                            ("candidate_policy.selection_guidance",
                             "selection_guidance")):
        try:
            sha = controller.surface_precondition_sha(parent, template)
        except Exception:  # noqa: BLE001
            continue
        add(template, template, operation="PATCH",
            precondition={"kind": "SHA", "sha": sha},
            current_value=policy.get(field))

    return tuple(entries)


def cause_for_surface(surface_id: str) -> str:
    """The registered cause for the surface the model chose."""
    target = str(surface_id)
    if target.startswith("skill_library.entries/"):
        if target.endswith(".body"):
            return CAUSE_FOR_TEMPLATE["skill_library.entries/{skill_id}.body"]
        if target.endswith(".observable_applicability"):
            return CAUSE_FOR_TEMPLATE[
                "skill_library.entries/{skill_id}.observable_applicability"]
        return CAUSE_FOR_TEMPLATE["skill_library.entries/{skill_id}"]
    if target.startswith("bootstrap_skills.entries/") and target.endswith(".body"):
        return CAUSE_FOR_TEMPLATE["bootstrap_skills.entries/{skill_id}.body"]
    if target in CAUSE_FOR_TEMPLATE:
        return CAUSE_FOR_TEMPLATE[target]
    raise EditRefused("surface outside this boundary's whitelist: %s" % target)


def edit_preflight(manifest: Any, catalog: Sequence[Mapping[str, Any]] = ()
                   ) -> None:
    """Fail closed on anything outside the four authorised objects."""
    target = str(getattr(manifest, "target_surface_id", "") or "")
    operation = str(getattr(getattr(manifest, "operation", None), "value",
                            getattr(manifest, "operation", None)))
    for forbidden in FORBIDDEN_TEMPLATES:
        stem = forbidden.replace("{skill_id}", "").replace("{memory_id}", "")
        if target == forbidden or (stem and target.endswith(stem.lstrip("/"))
                                   and stem not in (".body",)):
            raise EditRefused("surface is not editable in this package: %s"
                              % target)
    cause = cause_for_surface(target)  # raises for anything unlisted

    if operation == "ADD":
        if not target.startswith("skill_library.entries/"):
            raise EditRefused("ADD is authorised only for a new guidance entry")
        value = dict(getattr(manifest, "new_value", None) or {})
        if str(value.get("skill_kind")) != "capability":
            raise EditRefused("the new entry must be a capability entry", form=True)
        if not _is_guidance_body(value.get("body")):
            raise EditRefused(
                "a frozen program in the body would hand one replacement "
                "program to the whole group; guidance prose only")
        guards = dict(value.get("risk_guards") or {})
        if dict(guards.get("authority") or {}).get("supplies_candidates") is True:
            raise EditRefused("candidate-supply authority is not granted here")
        if value.get("serving_scope"):
            raise EditRefused(
                "a serving scope on a guidance entry would re-broadcast it "
                "past the sequence the decision belongs to")
        return

    if operation != "PATCH":
        raise EditRefused("only ADD and PATCH are authorised: %s" % operation, form=True)

    if getattr(manifest, "new_value", None):
        raise EditRefused("a PATCH surface cannot carry a deployable entry", form=True)
    patch = dict(getattr(manifest, "minimal_patch", None) or {})
    if "value" not in patch:
        raise EditRefused("PATCH requires minimal_patch.value", form=True)
    value = patch["value"]
    if cause == "SCOPE_OVERREACH":
        if not isinstance(value, Mapping):
            raise EditRefused("an applicability PATCH must be an object", form=True)
    else:
        if not isinstance(value, str) or not value.strip():
            raise EditRefused("a text PATCH must be a non-empty string", form=True)
        if not _is_guidance_body(value):
            raise EditRefused(
                "a frozen program in a guidance surface would hand one "
                "replacement program to every sequence that reads it")
    if catalog:
        offered = {str(row.get("surface_id")) for row in catalog}
        if target not in offered and "{skill_id}" not in target:
            raise EditRefused("surface was not offered in this catalog: %s"
                              % target)


# ---------------------------------------------------------------------------
# the card Slow reads
# ---------------------------------------------------------------------------

def evidence_card(group: Mapping[str, Any], *, pattern_id: str,
                  binding: FeatureBinding,
                  vocabulary: Sequence[str] = (),
                  decision_records: Sequence[Mapping[str, Any]] = (),
                  ) -> dict[str, Any]:
    """One group, its matched successes, and what the decisions actually did.

    Both faces are here: for a decision that passed Support and lost on the
    delayed face, the delayed number *is* the failure.  What is withheld is a
    whole later unit -- the one the two arms are then run on.
    """
    by_episode = {str(row.get("episode_id")): row
                  for row in decision_records if row.get("episode_id")}

    def trace_of(row: Mapping[str, Any]) -> dict[str, Any]:
        record = by_episode.get(str(row.get("episode_id"))) or {}
        return {
            "candidates_proposed": record.get("candidate_programs") or {},
            "probed": record.get("probe_programs") or [],
            "chosen": record.get("deployed_label"),
            "chosen_via": record.get("deployed_via"),
            "retrieved_skill_ids": record.get("retrieved_skill_ids") or [],
            "risk_refusals": record.get("risk_refusals"),
            "action_facts": record.get("action_facts") or "UNKNOWN",
        }

    def case(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "series_uid": row.get("series_uid"),
            "window_origin": row.get("origin"),
            "program": row.get("program_label_with_params")
                       or row.get("program_label"),
            "program_steps": row.get("program_steps"),
            "support_gain": row.get("support_gain"),
            "delayed_gain": row.get("delayed_gain"),
            "symptom": row.get("symptom"),
            "public_features_at_this_window": row.get("public_features"),
            "what_the_decision_did": trace_of(row),
        }

    successes = group["contrast_cases"]["positive"]
    return {
        "pattern_id": pattern_id,
        "observable_signature": dict(
            group["members"][0]["public_features"] if group["members"] else {}),
        "what_happened": (
            "%d per-sequence decisions ran %s and came out %s on their own "
            "evaluated sequence.  Each decision was taken for one sequence at "
            "one window, from that sequence's own visible context.  They are "
            "collected here because the typed workflow and the response sign "
            "match -- that is an organisation of the material, not a finding "
            "that they share one cause."
            % (group["member_count"], group["workflow"], group["sign"])),
        "failing_decisions": [case(row) for row in group["members"]],
        "matched_successes": [case(row) for row in successes],
        "matched_conflicts": [case(row) for row in
                              group["contrast_cases"]["conflict"]],
        "visible_pattern_spread": {
            "of_the_failing_decisions": group["visible_pattern_spread"],
            "of_the_matched_successes":
                group["visible_pattern_spread_of_the_matched_successes"],
            "how_to_read_it": (
                "the same deployment-visible features on both sides, each "
                "taken at the decision's own window.  Overlapping ranges "
                "separate nothing; disjoint ranges are a candidate condition. "
                "Neither is proof."),
        },
        "fault_labels_seen": {
            "selectable": group["fault"]["selectable_fault_types"],
            "guard_if_none": group["fault"]["guard_if_none"],
            "status": ("a statistical label over mechanical evidence.  It is a "
                       "hint about what the receipts look like.  It does not "
                       "decide which surface you may edit and it is not a "
                       "confirmed cause."),
        },
        "how_to_read_action_facts": (
            "what_the_decision_did.action_facts says how many points the "
            "program actually moved.  serving_window is this series' own "
            "window under this program; training_material is the SHARED "
            "training material for that (unit, face, program) and every "
            "series served by that model shares it -- it is not a property of "
            "the evaluated series.  A count is an action, not an effect: it "
            "is not a causal contribution and not a probability of harm.  "
            "UNKNOWN means not measured, never zero."),
        "unknowns_are_unknown": (
            "delayed_gain 'UNKNOWN' means the reading was never taken.  It is "
            "not zero and must not be reasoned about as a zero."),
        "observable_feature_vocabulary": list(vocabulary),
        "what_each_feature_is_measured_over": FEATURE_SEMANTICS,
        "how_to_use_the_feature_semantics": (
            "each entry gives the formula and the window it is computed on.  "
            "Three scopes are different objects and the vocabulary only "
            "covers the first: the feature window is the whole observed "
            "history; the serving window is its last 192 points and is the "
            "only thing the Consumer sees; the modified region is whatever "
            "the program actually changes at execution time, and no feature "
            "here measures it.  A threshold you write is therefore a "
            "statement about the whole history, not about the window that "
            "will be served and not about the region that will be changed.  "
            "Nothing here tells you how a value moves from one window to the "
            "next -- that is not recorded and you should not assume it."),
        "applicability_grammar": applicability_grammar(),
        "traceability_is_not_a_condition": (
            "series_uid and window_origin are here so you can trace a case "
            "back.  They are NOT part of the observable feature vocabulary "
            "and an applicability condition cannot name an entity, a dataset "
            "or a time position -- only the public features listed above."),
        "what_you_may_write": (
            "Exactly one edit, on exactly one surface from "
            "writable_surface_catalog.  Say four things: (1) which facts on "
            "this card you are relying on; (2) what behaviour of the Harness "
            "you suspect is wrong, as a mechanism hypothesis you accept may "
            "be wrong; (3) the one change; (4) what you expect to change in "
            "behaviour and effect, and what result would refute you.  Points "
            "(1) and (2) go in the text you write; (4) goes in "
            "predicted_agent_behavior_change, predicted_data_effect and "
            "falsification_condition.  Where behaviour still works, say what "
            "should be kept.  A provisional explanation is acceptable; you "
            "are not required to prove a unique cause, to explain the largest "
            "group, or to cover most of the failures.  If the public evidence "
            "supports no edit, return the no_proposal envelope."),
        "what_you_may_not_write": [
            "any surface not in writable_surface_catalog",
            "the string 'Frozen program steps:' anywhere -- a fixed program "
            "handed to every sequence is exactly what this boundary excludes; "
            "Fast still forms and verifies each sequence's own program",
            "candidate-supply authority, a serving scope, risk guards, "
            "verification rules, retrieval top_k or candidate slot counts",
            "an applicability condition naming a series, a dataset or a "
            "course position",
        ],
        "how_to_fill_the_manifest": {
            "one_surface_only": (
                "copy surface_id, operation, surface_precondition and "
                "dependency_precondition_shas verbatim from the one "
                "writable_surface_catalog entry you chose"),
            "ADD": {
                "new_value.skill_kind": "capability",
                "new_value.skill_id": ("a new lowercase id, not one in "
                                       "existing_entry_inventory; put the "
                                       "same id in target_surface_id"),
                "new_value.body": "the guidance prose",
                "new_value.allowed_tools": "[]",
                "observable_applicability": (
                    "the condition, in applicability_grammar's closed leaf "
                    "form; the same object in both "
                    "edit_manifest.observable_applicability and "
                    "new_value.observable_applicability"),
            },
            "PATCH": {
                "minimal_patch.value": (
                    "the COMPLETE replacement value for that surface -- a "
                    "PATCH replaces the whole field, so include everything "
                    "you want to keep from current_value.  It must differ "
                    "from current_value."),
                "observable_applicability": (
                    "null for a text surface; for "
                    "skill_library.entries/<id>.observable_applicability the "
                    "replacement condition object goes in minimal_patch.value"),
            },
            "predicted_agent_behavior_change": {
                "rule": ("a closed vocabulary.  Legal values: "
                         "'retrieve_skill:<a skill_id that exists after your "
                         "edit>', 'supply_effect_distinct', "
                         "'choose_candidate_kind:identity', "
                         "'choose_candidate_kind:program', "
                         "'identity_retained', "
                         "'effective_view_unchanged_out_of_scope', "
                         "'scope_modified_fraction<=0.NN', "
                         "'localization_iou>=0.NN', or a canonical operator "
                         "name.  Nothing else validates."),
            },
        },
    }


# ---------------------------------------------------------------------------
# proposing and applying
# ---------------------------------------------------------------------------

def _one_proposal(slow_agent: Any, card: Mapping[str, Any],
                  catalog: Sequence[Mapping[str, Any]], snapshot: Any,
                  task_context: Any) -> dict[str, Any]:
    def preflight(manifest: Any) -> None:
        edit_preflight(manifest, catalog)

    try:
        manifest = slow_agent.propose_edit(card, list(catalog), snapshot,
                                           manifest_preflight=preflight,
                                           task_context=task_context)
    except EditRefused as exc:
        return {"proposed": False,
                "why": ("REFUSED_BY_RUNTIME_FORM" if getattr(exc, "form", False)
                        else "REFUSED_BY_RUNTIME"),
                "detail": str(exc)[:300]}
    except Exception as exc:  # noqa: BLE001 - recorded, never hidden
        detail = "%s: %s" % (type(exc).__name__, str(exc)[:400])
        return {"proposed": False, "why": _fault_kind(detail),
                "detail": detail}
    if manifest is None:
        return {"proposed": False, "why": "SLOW_ABSTAINED",
                "detail": str(slow_agent.last_no_proposal_reason or "")[:300]}
    value = dict(getattr(manifest, "new_value", None) or {})
    patch = dict(getattr(manifest, "minimal_patch", None) or {})
    target = str(manifest.target_surface_id)
    return {
        "proposed": True,
        "manifest": manifest,
        "target_surface_id": target,
        "operation": str(getattr(manifest.operation, "value",
                                 manifest.operation)),
        "derived_cause": cause_for_surface(target),
        "skill_id": str(value.get("skill_id") or ""),
        "written_text": (str(value.get("body")) if value
                         else patch.get("value")),
        "observable_applicability": dict(
            value.get("observable_applicability")
            or getattr(manifest, "observable_applicability", None) or {}),
        "predicted_agent_behavior_change": list(
            getattr(manifest, "predicted_agent_behavior_change", ()) or ()),
        "predicted_data_effect": list(
            getattr(manifest, "predicted_data_effect", ()) or ()),
        "falsification_condition": list(
            getattr(manifest, "falsification_condition", ()) or ()),
    }


#: An account or permission error is not the model saying anything -- the
#: request never reached a model -- so it must not be recorded as an
#: abstention and must not be retried.  DEV-SEQ-2's first g1 attempt hit
#: ``402 Insufficient Balance`` at the boundary; the old classifier called it
#: SLOW_STAGE_FAULT, retried it once, and the run recorded NO_PROPOSAL, which
#: reads as "Slow declined to propose" and is false.
_ACCOUNT_MARKERS = (
    "insufficient balance", "insufficient_quota", "insufficient quota",
    "payment required", "error code: 402", "error code: 401",
    "invalid_api_key", "not enough available money", "quota exceeded",
    "billing",
)

#: A relay that reports an upstream failure of its own is not the account
#: saying no.  DEV-SEQ-3 g1 hit ``403 bad_response_status_code / openai_error``
#: from the relay while the same key answered a small request seconds later
#: and a 70KB payload seconds after that -- an upstream hiccup wrapped in a
#: 403, not a permission decision.  Classifying it as an account fault made it
#: non-retryable and ended the boundary on a transport blip.  These go through
#: the same bounded single retry as a schema fault; genuine account markers
#: above still stand and are never retried.
_TRANSPORT_MARKERS = (
    "bad_response_status_code", "openai_error", "upstream", "bad gateway",
    "service unavailable", "error code: 502", "error code: 503",
    "error code: 504", "timeout", "connection",
)


def _fault_kind(detail: str) -> str:
    lowered = str(detail).lower()
    if any(marker in lowered for marker in _ACCOUNT_MARKERS):
        return "ACCOUNT_OR_PERMISSION_FAULT"
    if any(marker in lowered for marker in _TRANSPORT_MARKERS):
        return "TRANSPORT_TRANSIENT_FAULT"
    return "SLOW_STAGE_FAULT"


#: A protocol/schema fault is an instrument failure -- the model returned JSON
#: the closed contract rejects -- so a fresh session may retry it after being
#: told the concrete error.  A legitimate abstention, a content refusal and an
#: account fault all stand: the first two are answers, the third is a
#: configuration receipt that blind retrying cannot fix.
RETRYABLE = ("SLOW_STAGE_FAULT", "TRANSPORT_TRANSIENT_FAULT",
             "REFUSED_BY_RUNTIME_FORM")


def propose_update(*, slow_factory: Any, card: Mapping[str, Any],
                   catalog: Sequence[Mapping[str, Any]], snapshot: Any,
                   task_context: Any = None,
                   attempts: int = 2) -> dict[str, Any]:
    """The real Slow stage.  Returns the manifest, or why there is none."""
    log: list[dict[str, Any]] = []
    outcome: dict[str, Any] = {"proposed": False, "why": "NO_ATTEMPT_MADE"}
    card = dict(card)
    for attempt in range(1, int(attempts) + 1):
        outcome = _one_proposal(slow_factory(), card, catalog, snapshot,
                                task_context)
        log.append({"attempt": attempt,
                    "proposed": bool(outcome.get("proposed")),
                    "why": outcome.get("why"),
                    "detail": outcome.get("detail")})
        if outcome.get("proposed") or outcome.get("why") not in RETRYABLE:
            break
        card["previous_attempt_was_rejected_with"] = str(
            outcome.get("detail") or "")[:400]
    outcome["attempts"] = log
    outcome["retry_rule"] = (
        "only a protocol/schema fault is retried, at most once, and the "
        "concrete error is handed back; an abstention and a content refusal "
        "both stand")
    return outcome


def apply_update(*, controller: Any, store: Any, snapshot: Any,
                 manifest: Any, catalog: Sequence[Mapping[str, Any]] = ()
                 ) -> dict[str, Any]:
    """Compile into an isolated fork.  The parent is kept, nothing activates."""
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (  # noqa: PLC0415
        _resolve_apply_manifest,
    )
    try:
        edit_preflight(manifest, catalog)
        cause = cause_for_surface(str(manifest.target_surface_id))
        resolved = _resolve_apply_manifest(manifest, snapshot)
        receipt = controller.apply_to_fork(store.materialize(snapshot),
                                           resolved, confirmed_cause=cause)
    except Exception as exc:  # noqa: BLE001 - recorded, never relaxed
        return {"applied": False,
                "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
    return {"applied": True,
            "snapshot": receipt.candidate_snapshot.snapshot,
            "confirmed_cause": cause,
            "runtime_bundle_sha": receipt.candidate_runtime_bundle_sha,
            "parent_runtime_bundle_sha": receipt.parent_runtime_bundle_sha,
            "target_surface_id": receipt.target_surface_id}


# ---------------------------------------------------------------------------
# what each visible feature is actually measured over
# ---------------------------------------------------------------------------

#: What each visible feature is, and what it is computed over.
#:
#: Facts only.  Every entry gives the formula as the extractor computes it
#: (``runtime/public_features.py``) and the window that formula runs on.  No
#: entry claims how a value moves between windows: DEV-SEQ-2 asserted that
#: some of these "decay" and others are "monotone non-decreasing" as the
#: course advances, and only one of those claims was ever measured (the
#: longest-missing-run ratio, on one series).  A measurement on one series is
#: not a property of a feature, so the claims are gone rather than repeated.
#:
#: Three scopes matter and they are not the same object:
#:
#:   feature window   ``request.values`` = ``values[:origin]`` -- the whole
#:                    observed history up to this decision's origin.  Every
#:                    feature below is computed on this, and recomputed from
#:                    scratch at every decision; nothing is carried over.
#:   serving window   ``values[origin-192:origin]`` -- the last 192 points.
#:                    This is the only thing the Consumer is given at serving
#:                    time.  At origin 1656 the feature window is 1656 points
#:                    and the serving window is 192, so a whole-history
#:                    feature is NOT a feature of the serving window.
#:   modified region  whatever the chosen program actually changes when it
#:                    runs on the serving window, measured against the
#:                    linear-integrity baseline.  It is decided at execution
#:                    time.  **No feature in this vocabulary measures it.**
FEATURE_WINDOW = "values[:origin] -- the whole observed history"
SERVING_WINDOW_POINTS = 192

FEATURE_SEMANTICS: dict[str, dict[str, str]] = {
    "_scopes": {
        "feature_window": (
            "values[:origin], the whole observed history up to this "
            "decision's origin; every feature below is computed on it and "
            "recomputed from scratch at every decision"),
        "serving_window": (
            "values[origin-192:origin], the last 192 points -- the only "
            "window the Consumer is given at serving time.  A feature "
            "computed over the whole history is not a feature of this window"),
        "modified_region": (
            "what the chosen program actually changes on the serving window, "
            "against the linear-integrity baseline, decided at execution "
            "time.  No feature in this vocabulary measures it"),
    },
    "task_kind": {
        "formula": "the Task type of this decision",
        "computed_over": "not a data quantity",
    },
    "missing_fraction": {
        "formula": "mean(~isfinite(values))",
        "computed_over": "the feature window",
        "denominator": "number of points in the feature window",
    },
    "longest_missing_run_fraction": {
        "formula": ("length of the longest run of consecutive missing "
                    "points, divided by values.size"),
        "computed_over": "the feature window",
        "denominator": "number of points in the feature window",
    },
    "local_robust_z_peak": {
        "formula": ("max over the window of |filled - median| / scale, where "
                    "median and scale are robust statistics of the same "
                    "window"),
        "computed_over": "the feature window",
        "denominator": "a robust scale of the feature window",
    },
    "estimated_region_start_fraction": {
        "formula": ("index of the first point of the union of the missing, "
                    "outlier and level regions, divided by values.size"),
        "computed_over": "the feature window",
        "denominator": "number of points in the feature window",
    },
    "estimated_region_end_fraction": {
        "formula": ("one past the last point of that same union, divided by "
                    "values.size"),
        "computed_over": "the feature window",
        "denominator": "number of points in the feature window",
    },
    "level_region_fraction": {
        "formula": "mean(level_mask)",
        "computed_over": "the feature window",
        "denominator": "number of points in the feature window",
    },
    "level_region_end_fraction": {
        "formula": ("one past the last point of the level mask, divided by "
                    "values.size"),
        "computed_over": "the feature window",
        "denominator": "number of points in the feature window",
    },
    "outlier_region_end_fraction": {
        "formula": ("one past the last point of the expanded outlier region, "
                    "divided by values.size"),
        "computed_over": "the feature window",
        "denominator": "number of points in the feature window",
    },
    "post_shift_support_sufficient": {
        "formula": ("(1 - estimated_region_end_fraction) * 240 >= 24, as a "
                    "boolean"),
        "computed_over": ("a fraction of the feature window, rescaled by the "
                          "fixed constant 240"),
        "note": ("the 240 is a fixed downstream-window constant, not the "
                 "feature window's own length"),
    },
    "level_only_post_shift_support_sufficient": {
        "formula": "the same test applied to level_region_end_fraction",
        "computed_over": "as above",
    },
    "level_excursion_score": {
        "formula": "the level-candidate score of the window",
        "computed_over": "the feature window",
    },
    "estimated_level_offset": {
        "formula": ("median(filled[level_mask]) - reference, in the data's "
                    "own units"),
        "computed_over": "the feature window",
        "denominator": "none; this is not a fraction",
    },
    "period_change_score": {
        "formula": "period-change score from the period summary",
        "computed_over": "the feature window, after removing the level offset",
    },
    "period_reliability": {
        "formula": "reliability term from the same period summary",
        "computed_over": "the feature window, after removing the level offset",
    },
    "period_evidence_status": {
        "formula": "categorical status from the same period summary",
        "computed_over": "the feature window, after removing the level offset",
    },
    "period_repair_available": {
        "formula": "always False in this extractor",
        "computed_over": "not a data quantity",
    },
    "imputation_probe_direction": {
        "formula": "direction reported by the fixed imputation probe panel",
        "computed_over": ("the fixed probe panel, not recomputed from the "
                          "window"),
    },
    "clipping_probe_direction": {
        "formula": "direction reported by the fixed clipping probe panel",
        "computed_over": "the fixed probe panel",
    },
    "denoising_probe_direction": {
        "formula": "direction reported by the fixed denoising probe panel",
        "computed_over": "the fixed probe panel",
    },
    "level_probe_direction": {
        "formula": ("direction reported by the fixed level-correction probe "
                    "panel"),
        "computed_over": "the fixed probe panel",
    },
}


# ---------------------------------------------------------------------------
# action facts: what the program actually changed, in points
# ---------------------------------------------------------------------------

#: Memo for the training-side count, which is a property of the (unit, face,
#: program) triple and not of any one evaluated series.
_TRAINING_ACTION_MEMO: dict[tuple, dict[str, Any]] = {}


def _compiled_program(ctx: Any, origin: int, steps: Sequence[tuple]) -> Any:
    from evaluation.main_protocol_p4 import run_hec1 as runner  # noqa: PLC0415
    at = ctx.cell_at(int(origin))
    config = runner.forecast_p4._config(int(origin))
    roster = at.roster(runner.FACE)
    executor = runner._executor(roster, at.values, config)
    return at, config, roster, executor._compiled(tuple(steps))


def action_facts(ctx: Any, origin: int, steps: Sequence[tuple] | None,
                 uid: str) -> dict[str, Any]:
    """How many points the program moved, on the serving window and in training.

    Both numbers are produced by replaying the evaluator's own ``_prepare`` on
    windows that are already legally open at this origin.  Nothing is re-fitted
    and no Consumer time is spent: ``_prepare`` is the same function the
    evaluation path calls, the baseline is the same ``_linear_integrity``, and
    the tolerance is the same ``isclose(..., equal_nan=True)``.  The serving
    count is the one the evaluator computes and discards; the training count is
    the same quantity the evaluator reports as ``behavior_point_count``.

    Semantics, which the numbers do not carry on their own:

    * the serving count belongs to **this** series at **this** window under
      **this** program and its parameters; Support and delayed are separate
      readings and get separate counts;
    * the training count is a property of the **shared training material** for
      this (unit, face, program).  Every evaluated series served by that model
      shares it.  It is not a property of the evaluated series and must not be
      written as one;
    * a count is an action, not an effect.  It is not a causal contribution and
      not a probability of harm;
    * an unavailable count stays UNKNOWN.  It is never zero.
    """
    import numpy as np  # noqa: PLC0415

    from evaluation.main_protocol_p4 import (  # noqa: PLC0415
        scoped_serving_evaluator as scoped,
    )

    out: dict[str, Any] = {
        "series_uid": str(uid),
        "origin": int(origin),
        "program": ps.program_label_with_params(ps.normalise_steps(steps or ())),
        "baseline": "forecast_runtime._linear_integrity, the evaluator's own",
        "tolerance": "numpy isclose(..., equal_nan=True), the evaluator's own",
        "counts_are_actions_not_effects": (
            "a moved-point count says what the program did, not what it "
            "caused.  It is not a causal contribution and not a probability "
            "of harm"),
    }
    normalised = ps.normalise_steps(steps or ())
    if not normalised:
        out["serving_window"] = {
            "modified_points": 0,
            "total_points": int(scoped.CONTEXT_LENGTH),
            "why": ("identity: the raw pipeline ran, so nothing was moved off "
                    "the linear-integrity baseline.  This 0 is measured, not "
                    "filled in"),
        }
        out["training_material"] = {
            "modified_points": 0,
            "total_points": "UNKNOWN",
            "why": "identity: the program model is the raw model",
            "whose_property": ("the shared training material for this unit "
                               "and face, not this evaluated series"),
        }
        return out

    try:
        at, config, roster, compiled = _compiled_program(ctx, origin, normalised)
    except Exception as exc:  # noqa: BLE001 - recorded, never guessed
        out["unavailable"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        out["serving_window"] = {"modified_points": "UNKNOWN",
                                 "total_points": "UNKNOWN"}
        out["training_material"] = {"modified_points": "UNKNOWN",
                                    "total_points": "UNKNOWN"}
        return out

    # ---- this series' own serving window --------------------------------
    raw = np.asarray(at.values[str(uid)], dtype=np.float64)
    window = raw[int(origin) - scoped.CONTEXT_LENGTH:int(origin)]
    try:
        _served, moved, _trace = scoped._prepare(window, compiled)
        out["serving_window"] = {
            "modified_points": int(moved),
            "total_points": int(window.size),
            "modified_fraction": (round(float(moved) / float(window.size), 6)
                                  if window.size else "UNKNOWN"),
            "whose_property": ("this series, this window, this program and "
                               "these parameters"),
        }
    except Exception as exc:  # noqa: BLE001
        out["serving_window"] = {"modified_points": "UNKNOWN",
                                 "total_points": int(window.size),
                                 "why": "%s: %s" % (type(exc).__name__,
                                                    str(exc)[:160])}

    # ---- the shared training material -----------------------------------
    key = (json.dumps(ctx.unit, sort_keys=True, default=str), int(origin),
           ps.program_signature(normalised))
    memo = _TRAINING_ACTION_MEMO.get(key)
    if memo is None:
        try:
            windows = scoped._training_windows(roster, at.values, config,
                                               int(origin))
            moved_total = 0
            point_total = 0
            for train_window in windows:
                _prepared, train_moved, _t = scoped._prepare(train_window,
                                                             compiled)
                moved_total += int(train_moved)
                point_total += int(train_window.size)
            memo = {"modified_points": moved_total,
                    "total_points": point_total,
                    "training_windows": len(windows),
                    "modified_fraction": (
                        round(moved_total / point_total, 6)
                        if point_total else "UNKNOWN")}
        except Exception as exc:  # noqa: BLE001
            memo = {"modified_points": "UNKNOWN", "total_points": "UNKNOWN",
                    "why": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
        _TRAINING_ACTION_MEMO[key] = memo
    out["training_material"] = dict(memo)
    out["training_material"]["whose_property"] = (
        "the shared training material for this (unit, face, program).  Every "
        "evaluated series served by that model shares this number; it is not "
        "a property of this evaluated series and must not be read as one")
    return out


# ---------------------------------------------------------------------------
# exposure: did the edit actually reach the decisions we are about to pay for?
# ---------------------------------------------------------------------------

def fast_view_bytes(snapshot: Any, features: Mapping[str, Any]) -> str:
    """Exactly what Fast resolves for one sequence, as bytes, role=fast."""
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: PLC0415
        resolve_harness_view,
    )
    view = resolve_harness_view(snapshot, dict(features), role="fast")
    return json.dumps({
        "instruction": view.instruction,
        "skills": [{"skill_id": s.skill_id, "body": s.body} for s in view.skills],
        "controls": json.loads(json.dumps(view.controls, default=dict)),
    }, ensure_ascii=False, sort_keys=True, default=str)


def exposure_check(*, parent: Any, candidate: Any, probe: str,
                   windows: Mapping[str, Mapping[str, Mapping[str, Any]]],
                   skill_id: str = "") -> dict[str, Any]:
    """Is the edit visible to the decisions the arms are about to make?

    Costs no Consumer fits and no LLM calls: it resolves the same view Fast
    resolves, over the same window-bound public features the arms will use,
    and looks for the edited text in the bytes.

    This exists because DEV-SEQ-2's g1 compiled a legal candidate whose
    condition matched **none** of the sequences in any remaining registered
    window, so every arm decision would have run under the parent knowledge
    with the candidate present but never retrieved.  Paying for hours of
    paired decisions to measure a guidance that is never presented is not a
    test of the guidance.  It is not a coverage gate and it does not ask the
    predicate to explain most failures -- it asks only whether the thing
    about to be measured is on screen at all.
    """
    from SelfEvolvingHarnessTS.methods.ttha import ordering_card as oc

    rows: list[dict[str, Any]] = []
    for position, cards in sorted(windows.items(), key=lambda kv: str(kv[0])):
        for uid, features in sorted((cards or {}).items()):
            in_candidate = probe in fast_view_bytes(candidate, features)
            in_parent = probe in fast_view_bytes(parent, features)
            rows.append({"position": str(position), "series_uid": str(uid),
                         "edit_is_in_the_candidate_view": bool(in_candidate),
                         "edit_is_in_the_parent_view": bool(in_parent)})
    exposed = [row for row in rows if row["edit_is_in_the_candidate_view"]]
    leaked = [row for row in rows if row["edit_is_in_the_parent_view"]]

    # the one other path a Skill body can take into a Fast round: the
    # ordering-card selector reads the active snapshot directly rather than
    # through resolve_harness_view.
    ordering = None
    if skill_id:
        entry = next((sk for sk in getattr(candidate, "skills", ())
                      if str(sk.skill_id) == str(skill_id)), None)
        ordering = {
            "entry_found_in_candidate": entry is not None,
            "is_an_ordering_card": bool(entry is not None
                                        and oc.is_ordering_card(entry)),
            "why_it_matters": (
                "_select_ordering_card reads the snapshot directly instead of "
                "going through resolve_harness_view, so it is the one other "
                "route a Skill body can take into a Fast round.  It also "
                "requires the same observable_applicability to pass, so a "
                "predicate that matches nothing closes this route too"),
        }
    return {
        "decisions_checked": len(rows),
        "decisions_the_edit_reaches": len(exposed),
        "exposure_fraction": (round(len(exposed) / len(rows), 4) if rows
                              else 0.0),
        "the_edit_leaks_into_the_parent_arm": len(leaked),
        "per_decision": rows,
        "ordering_card_route": ordering,
        "cost": "zero Consumer fits, zero LLM calls",
        "what_this_is_not": (
            "not a coverage gate.  It does not require the condition to "
            "explain most failures or to match a majority of anything.  It "
            "asks only whether the edit is presented to the decisions that "
            "are about to be paid for; a modification that is never on "
            "screen cannot be measured by running more of them"),
    }


# ---------------------------------------------------------------------------
# the old three lines, kept as diagnostics only
# ---------------------------------------------------------------------------

def legacy_lines(*, applicability: Mapping[str, Any],
                 group: Mapping[str, Any], binding: FeatureBinding,
                 withheld_population: Sequence[str],
                 withheld_origin: int,
                 withheld_features: Mapping[str, Mapping[str, Any]],
                 ) -> dict[str, Any]:
    """DEV-SEQ-1's L1/L2, recomputed with correct window binding.

    Reported, never acted on.  L3 is not recomputed here: it costs Consumer
    fits to re-read old programs, and this package spends those fits on the
    two real Fast arms instead, which is the reading that actually decides.
    """
    if not applicability:
        return {"applicable": False,
                "why": "this edit carries no applicability predicate"}
    population = [str(uid) for uid in withheld_population]
    matched = [uid for uid in population
               if know1._matches(applicability, withheld_features.get(uid) or {})]
    l1 = 1 <= len(matched) <= max(len(population) - 1, 0)

    failing_windows = [(row["series_uid"], row["origin"])
                       for row in group["members"]
                       if row["series_uid"] and row["origin"] is not None]
    covered = [key for key in sorted(set(failing_windows))
               if know1._matches(applicability, binding.get(*key))]
    distinct = sorted(set(failing_windows))
    l2 = bool(distinct) and len(covered) * 2 > len(distinct)
    return {
        "applicable": True,
        "L1_the_predicate_is_a_condition": bool(l1),
        "L1_detail": {"withheld_population": len(population),
                      "matched": matched},
        "L2_it_covers_the_evidence_it_cites": bool(l2),
        "L2_detail": {
            "failing_windows": [list(key) for key in distinct],
            "covered": [list(key) for key in covered],
            "counted_by": ("distinct (series_uid, window) decisions, each "
                           "tested against its own window's features -- "
                           "DEV-SEQ-1 tested every member against the first "
                           "window that series appeared in"),
        },
        "status": ("DIAGNOSTIC ONLY.  These lines describe the predicate's "
                   "coverage.  They do not gate adoption in this package, "
                   "because coverage is not the same question as whether the "
                   "guidance changes what Fast does and helps"),
    }


# ---------------------------------------------------------------------------
# adoption: decided by the two real Fast arms on the withheld unit
# ---------------------------------------------------------------------------

def paired_delayed(rows_a: Sequence[Mapping[str, Any]],
                   rows_b: Sequence[Mapping[str, Any]],
                   *, face: str = "delayed_gain") -> dict[str, Any]:
    """A vs B on exactly the same UID set and the same denominator."""
    a = {str(row["series_uid"]): row.get(face) for row in rows_a}
    b = {str(row["series_uid"]): row.get(face) for row in rows_b}
    population = sorted(set(a) | set(b))
    readable = [uid for uid in population
                if isinstance(a.get(uid), (int, float))
                and not isinstance(a.get(uid), bool)
                and isinstance(b.get(uid), (int, float))
                and not isinstance(b.get(uid), bool)]
    missing = [uid for uid in population if uid not in readable]
    complete = bool(population) and not missing and set(a) == set(b)
    mean_a = (sum(float(a[uid]) for uid in readable) / len(readable)
              if readable else None)
    mean_b = (sum(float(b[uid]) for uid in readable) / len(readable)
              if readable else None)
    return {
        "population": population,
        "population_size": len(population),
        "readable_pairs": readable,
        "missing_on_either_side": missing,
        "complete": complete,
        "mean_a": round(mean_a, 6) if mean_a is not None else "UNKNOWN",
        "mean_b": round(mean_b, 6) if mean_b is not None else "UNKNOWN",
        "paired_difference_b_minus_a": (
            round(mean_b - mean_a, 6)
            if (mean_a is not None and mean_b is not None) else "UNKNOWN"),
        "whole_population_difference": (
            round(mean_b - mean_a, 6)
            if (complete and mean_a is not None and mean_b is not None)
            else "UNKNOWN"),
        "coverage_of_the_readable_subset": (
            round(len(readable) / len(population), 4) if population else 0.0),
        "denominator_rule": ("the same UID set and the same denominator on "
                             "both sides; a missing reading on either side "
                             "makes the whole-population difference UNKNOWN "
                             "and the readable subset is reported separately, "
                             "never in its place"),
    }


def adoption_check(*, validation_a: Mapping[str, Any],
                   validation_b: Mapping[str, Any]) -> dict[str, Any]:
    """This package's development-level 'adoptable' mark.

    Three conditions, all on the existing constants:
      * the two readings are complete and comparable on the same population;
      * B beats A by MATERIAL on the paired delayed face;
      * B passes the existing authoritative gate on that unit.
    No constant and no risk denominator is moved for this, and passing it is
    not a production deployment permission.
    """
    pair = paired_delayed(validation_a["sequences"], validation_b["sequences"])
    gate = validation_b.get("unit_authoritative_gate") or {}
    gate_passes = bool(gate.get("passes")) if gate else False
    difference = pair["whole_population_difference"]
    beats = isinstance(difference, float) and difference >= MATERIAL
    lines = {
        "readings_complete_and_comparable": bool(pair["complete"]),
        "paired_delayed_gain_reaches_material": bool(beats),
        "candidate_passes_the_authoritative_gate_on_this_unit": gate_passes,
    }
    return {
        "paired": pair,
        "material_threshold": MATERIAL,
        "candidate_gate": gate,
        "lines": lines,
        "adoptable": all(lines.values()),
        "failed_lines": [name for name, ok in lines.items() if not ok],
        "registered_before_the_run": True,
        "what_it_is_not": (
            "a development-level adoption mark for this package.  It is not a "
            "production deployment permission, it does not grant a Shared "
            "Capability, and failing it does not end the candidate's "
            "follow-up: an unadopted candidate is still carried into the "
            "follow-up units as a diagnostic branch, so 'the guidance has no "
            "value' can be told apart from 'the adoption rule did not pass "
            "it'"),
    }


__all__ = [
    "CAUSE_FOR_TEMPLATE",
    "action_facts",
    "FEATURE_SEMANTICS",
    "exposure_check",
    "fast_view_bytes",
    "EditRefused",
    "FeatureBinding",
    "MeasurementLedger",
    "REACHES_FAST",
    "adoption_check",
    "apply_update",
    "bind_window_features",
    "build_catalog",
    "cause_for_surface",
    "edit_preflight",
    "evidence_card",
    "group_failures",
    "legacy_lines",
    "paired_delayed",
    "propose_update",
    "window_key",
]
