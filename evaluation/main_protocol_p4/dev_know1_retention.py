"""DEV-KNOW-1: the minimal knowledge-retention repair, and its two checks.

Why anything is repaired at all
-------------------------------
DEV-AUTO-3 located one mechanism precisely.  The updating arm's ``PATCH``
rewrote the K0 card's frozen body **in place and in one slot**, so a candidate
that later failed an authoritative delayed gate did not fall back to the older
body: the lifecycle revoked the whole card, the ancestor program went with it,
and the last three units of that arm had no card left to deploy.  One cell
(-0.595) paid back everything the update channel had earned.

The repair is therefore not "patch more carefully".  It is: **a verified new
program becomes a card of its own, beside the parent, through the ordinary
minting path the Fast lifecycle already owns.**  That path is
``method.handle_fast_winner`` -> ``EditOperation.ADD`` with
``surface_precondition={"kind": "ABSENT"}`` -> two-phase delayed approval ->
``online_loop.activate_approved``.  Arm B of DEV-AUTO-3 already ran exactly
that path and minted ``fast_winner_forecast_ridge_smase_outlier_iqr`` beside
K0.  So nothing here rebuilds a minting route; this module supplies only what
that route did not already have.

What this module adds, and nothing more
---------------------------------------
1.  ``duplicate_capability_card`` -- a **reading**, not a new rule.  "The same
    program must not get a second card" is already enforced by the mint that
    exists: ``_fast_winner_skill_id`` derives the id from the task scope and
    the winner's workflow signature, and the ADD carries
    ``surface_precondition={"kind": "ABSENT"}``, so a second card for the same
    program collides on the id and is refused before anything is written.  A
    guard was drafted for ``method.handle_fast_winner`` and then **removed**:
    ``method_contract`` is one of the harness's dependency shas, so editing
    that file changes every snapshot's ``runtime_bundle_sha`` and invalidates
    the locked h0 -- a new hash platform, which this package may not create.
    The predicate stays here so the run can *report* that no two cards hold
    the same knowledge, and the third check below exercises the refusal that
    is already in force.
2.  ``current_card`` -- an honest answer to "what does the card hold now".
    ``run_dev_auto1_skill_revision.current_skill_steps`` answers a missing card
    with ``ANCESTOR_PROGRAM``, which is the historical ancestor standing in for
    a card that is not there.  Nothing in this package may read that fallback,
    and ``card_or_refuse`` is what the runner calls instead.
3.  ``remove_cards`` -- the same store round-trip
    ``online_loop.revoke_deployed_skill`` uses (fork -> unlink the entry ->
    ``compile_snapshot`` -> materialize), lifted so it can take a *set* of
    skill ids.  This package uses it for exactly one thing: constructing the
    old-knowledge control arm at the fork by removing what the formation
    segment added.  It grants nothing and it revives nothing.
4.  The two behaviours the work order asks to be checked, executed against the
    real store and compiler rather than asserted from source text:
    ``a_child_card_coexists_with_its_parent`` and
    ``revoking_a_child_leaves_the_parent``.

What is deliberately *not* here
-------------------------------
No resurrection of a revoked ancestor, no zeroing of a Draft's revision or
verification counters, no route by which an ancestor skips the current unit's
Support, and no PATCH of any card at all.  Retention in this package is
additive or it does not happen.

Zero LLM calls and zero Consumer fits: everything below is store and compiler
work.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The card the whole course starts from.  Named here rather than imported so
#: the checks below can run without the course runner being importable.
K0_SKILL_ID = "fast_winner_forecast_ridge_smase_outlier_mad"

#: Closure recorded on Drafts that are still open when the formation segment
#: ends.  They are closed rather than deleted: the record -- its state, its
#: revision count, its verification count -- is what the runtime constraint
#: reads, and deleting records to restore a permission is the one move this
#: package is forbidden to make.  Closing removes them from
#: ``resupplied_programs_for_verification`` in *both* arms at the same instant,
#: so neither test arm inherits a pool of pre-loaded answers.
FORK_CLOSURE = "NOT_CARRIED_ACROSS_THE_FORK"


# ---------------------------------------------------------------------------
# reading the library honestly
# ---------------------------------------------------------------------------

def _frozen(value: Any) -> Any:
    """Mappings, proxies and tuples reduced to something comparable by value."""
    if isinstance(value, Mapping):
        return {str(k): _frozen(v)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_frozen(v) for v in value]
    return value


def capability_cards(snapshot: Any) -> dict[str, dict[str, Any]]:
    """Every capability card in a snapshot, by id, with what identifies it.

    Bootstrap procedures are excluded on purpose: they are the resident reading
    instructions, not knowledge this course can form or lose.
    """
    out: dict[str, dict[str, Any]] = {}
    for skill in list(getattr(snapshot, "skills", ()) or ()):
        kind = str(getattr(getattr(skill, "skill_kind", None), "value", ""))
        if kind != "capability":
            continue
        guards = dict(getattr(skill, "risk_guards", {}) or {})
        out[str(skill.skill_id)] = {
            "skill_id": str(skill.skill_id),
            "body": str(getattr(skill, "body", "") or ""),
            "observable_applicability": _frozen(
                getattr(skill, "observable_applicability", None)),
            "serving_scope": _frozen(getattr(skill, "serving_scope", None)),
            "requires_target_support": bool(
                guards.get("requires_target_support")),
            "risk_guards": _frozen(guards),
        }
    return out


def card_identity(*, body: str, applicability: Any,
                  serving_scope: Any) -> str:
    """The identity a second card may not repeat: program x where it applies.

    Deliberately not a hash of the whole entry.  Two cards differing only in a
    revision counter or a provenance note are the same knowledge; two cards
    differing in the frozen body, in the applicability that decides whether the
    card is even retrieved, or in the serving Scope that decides which series
    it treats, are not.
    """
    return json.dumps({"body": str(body or ""),
                       "observable_applicability": _frozen(applicability),
                       "serving_scope": _frozen(serving_scope)},
                      sort_keys=True, ensure_ascii=False, default=str)


def duplicate_capability_card(snapshot: Any, *, body: str,
                              applicability: Any,
                              serving_scope: Any) -> str | None:
    """The id of a card that already holds this program here, or ``None``."""
    target = card_identity(body=body, applicability=applicability,
                           serving_scope=serving_scope)
    for skill_id, row in capability_cards(snapshot).items():
        if card_identity(body=row["body"],
                         applicability=row["observable_applicability"],
                         serving_scope=row["serving_scope"]) == target:
            return skill_id
    return None


def current_card(snapshot: Any, skill_id: str = K0_SKILL_ID) -> dict[str, Any]:
    """What the card holds now, or a plain statement that it is not there.

    The steps come from Fast's own parser, so "what the card holds" is what the
    next retrieval would actually run -- not a re-derivation of it.
    """
    cards = capability_cards(snapshot)
    row = cards.get(str(skill_id))
    if row is None:
        return {
            "skill_id": str(skill_id),
            "present": False,
            "steps": None,
            "program": None,
            "why": ("no card with this id is in the active snapshot; the "
                    "historical ancestor is not offered in its place"),
            "capability_cards_present": sorted(cards),
        }
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: PLC0415
        _parse_frozen_steps,
    )
    parsed = _parse_frozen_steps(row["body"])
    steps = tuple((str(op), dict(params)) for op, params in (parsed or ()))
    return {
        "skill_id": str(skill_id),
        "present": True,
        "steps": steps,
        "program": ">".join(op for op, _p in steps) if steps else None,
        "body_unparseable": not steps,
        "requires_target_support": row["requires_target_support"],
        "serving_scope": row["serving_scope"],
        "capability_cards_present": sorted(cards),
    }


def card_or_refuse(snapshot: Any,
                   skill_id: str = K0_SKILL_ID) -> tuple[bool, dict[str, Any]]:
    """``(usable, report)``.  An absent or unparseable card is not usable.

    The caller is expected to stop rather than substitute anything: this is the
    one place where "there is no current card" has to stay a fact instead of
    quietly becoming a program.
    """
    report = current_card(snapshot, skill_id)
    usable = bool(report["present"]) and not report.get("body_unparseable")
    return usable, report


# ---------------------------------------------------------------------------
# adding a card beside its parent, and removing one without touching the rest
# ---------------------------------------------------------------------------

def add_card(*, controller: Any, store: Any, snapshot: Any, skill_id: str,
             steps: Sequence[tuple[str, Mapping[str, Any]]],
             applicability: Mapping[str, Any],
             serving_scope: Mapping[str, Any] | None = None,
             requires_target_support: bool = True) -> dict[str, Any]:
    """One card added beside whatever is already there, through the real path.

    The manifest is the shape ``method.handle_fast_winner`` builds -- ``ADD``,
    ``surface_precondition={"kind": "ABSENT"}``, a ``skill-entry/1`` value --
    and it goes through ``controller.apply_to_fork``, which validates the
    shape, checks the route authorisation, forks, applies and **compiles**.
    Used by the checks below; the course itself never calls it, because there
    the card is minted by the Fast lifecycle exactly as it always was.
    """
    from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: PLC0415
        EditManifest,
        EditOperation,
    )
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (  # noqa: PLC0415
        _resolve_apply_manifest,
    )
    body = "Frozen program steps: " + json.dumps(
        [{"op": op, "params": dict(params)} for op, params in steps])
    parent = store.materialize(snapshot)
    guards: dict[str, Any] = {"explicit_choice_required": True,
                              "observable_applicability_only": True,
                              "preserve_outside_candidate_region": True,
                              "single_surface_only": True}
    if requires_target_support:
        guards["requires_target_support"] = True
    value: dict[str, Any] = {
        "schema_version": "skill-entry/1",
        "skill_id": str(skill_id),
        "skill_kind": "capability",
        "revision": 1,
        "body": body,
        "observable_applicability": dict(applicability),
        "allowed_tools": [op for op, _p in steps],
        "risk_guards": guards,
    }
    if serving_scope:
        value["serving_scope"] = dict(serving_scope)
    manifest = EditManifest(
        edit_id=str(skill_id),
        base_harness_sha=parent.harness_content_sha,
        target_pattern_id="dev-know1-retention",
        target_surface_id="skill_library.entries/%s" % skill_id,
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value=value,
        observable_applicability=dict(applicability),
        patch_id=None,
        predicted_agent_behavior_change=("retrieve_skill:%s" % skill_id,),
        predicted_data_effect=("local_improvement",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
    )
    try:
        resolved = _resolve_apply_manifest(manifest, snapshot)
        receipt = controller.apply_to_fork(parent, resolved,
                                           confirmed_cause="SKILL_LIBRARY_GAP")
    except Exception as exc:  # noqa: BLE001 - recorded, never relaxed
        return {"applied": False, "skill_id": str(skill_id),
                "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
    return {"applied": True, "skill_id": str(skill_id),
            "snapshot": receipt.candidate_snapshot.snapshot,
            "runtime_bundle_sha": receipt.candidate_runtime_bundle_sha,
            "body": body}


def remove_cards(*, store: Any, snapshot: Any,
                 skill_ids: Sequence[str]) -> dict[str, Any]:
    """Remove named skill entries; leave every other entry byte-for-byte alone.

    The round-trip is ``online_loop.revoke_deployed_skill``'s: fork the
    materialised snapshot, unlink the entry files, recompile, materialise the
    result.  Lifting it to a set of ids changes nothing about how one removal
    works -- and that is the property the second check exercises.

    The active pointer is **not** moved here.  Whoever asked for the removal
    decides what to make active, which keeps this function unable to grant
    anything on its own.
    """
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: PLC0415
        compile_snapshot,
    )
    wanted = [str(s) for s in skill_ids]
    if not wanted:
        return {"removed": [], "not_found": [], "snapshot": snapshot,
                "files_unlinked": [], "unchanged": True,
                "skills_after": sorted(capability_cards(snapshot)),
                "runtime_bundle_sha": getattr(snapshot, "runtime_bundle_sha",
                                              None),
                "why": "nothing was asked to be removed"}
    parent = store.materialize(snapshot)
    fork = store.fork(parent, edit_id="know1_remove_%d" % len(wanted))
    removed: list[str] = []
    files: list[str] = []
    try:
        for sub in ("learned", "bootstrap"):
            directory = fork / "skills" / sub
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 - a file we cannot read is left
                    continue
                if isinstance(doc, dict) and str(doc.get("skill_id")) in wanted:
                    path.unlink()
                    removed.append(str(doc.get("skill_id")))
                    files.append(str(path.name))
        new_snapshot = compile_snapshot(fork, verify_lock=False)
        store.materialize(new_snapshot,
                          parent_sha=getattr(snapshot, "runtime_bundle_sha",
                                             None))
    finally:
        shutil.rmtree(fork, ignore_errors=True)
    return {
        "removed": sorted(removed),
        "not_found": sorted(set(wanted) - set(removed)),
        "files_unlinked": files,
        "snapshot": new_snapshot,
        "runtime_bundle_sha": new_snapshot.runtime_bundle_sha,
        "parent_runtime_bundle_sha": getattr(snapshot, "runtime_bundle_sha",
                                             None),
        "skills_after": sorted(capability_cards(new_snapshot)),
        "unchanged": False,
        "how": ("the same fork / unlink / compile / materialize round-trip "
                "online_loop.revoke_deployed_skill performs for one card"),
    }


# ---------------------------------------------------------------------------
# the two behaviours the work order asks to be checked
# ---------------------------------------------------------------------------

def _fixture(tag: str) -> dict[str, Any]:
    """K0's real snapshot in a throwaway store.  No fits, no model calls."""
    from evaluation.main_protocol_p4 import (  # noqa: PLC0415
        audit_m_r0_reachability as base,
    )
    from evaluation.main_protocol_p4 import (  # noqa: PLC0415
        run_dev_auto1_skill_revision as R,
    )
    from evaluation.main_protocol_p4 import (  # noqa: PLC0415
        run_m_r0d_forward_k1_outer_step as live,
    )
    from evaluation.main_protocol_p4 import run_source_line as v1runner  # noqa: PLC0415

    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    machinery = v1runner._machinery()
    directory, source = R._resolve_k0_snapshot(doc, state["k0"])
    if directory is None:
        raise RuntimeError("no readable K0 snapshot to check against")
    snapshot = machinery["compile_snapshot"](directory, verify_lock=False)
    root = Path(base.ROOT) / "_scratch" / "dev_know1_checks" / tag
    shutil.rmtree(root, ignore_errors=True)
    store = machinery["SnapshotStore"](root / "store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"]())
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    return {"snapshot": snapshot, "store": store, "controller": controller,
            "k0_snapshot_source": source}


#: The child used by both checks.  A two-step program, so its body cannot be
#: confused with the ancestor's, under the wide applicability a Fast winner
#: gets.
CHILD_SKILL_ID = "fast_winner_forecast_ridge_smase_know1_child"
CHILD_STEPS = (("winsorize", {}), ("outlier_mad", {}))
CHILD_APPLICABILITY = {"all": [{"feature": "task_kind", "op": "==",
                                "value": "forecast"}]}


def a_child_card_coexists_with_its_parent() -> dict[str, Any]:
    """Behaviour 1: the new program lands beside the parent, not over it."""
    fixture = _fixture("coexist")
    before = capability_cards(fixture["snapshot"])
    parent_before = before.get(K0_SKILL_ID)
    added = add_card(controller=fixture["controller"], store=fixture["store"],
                     snapshot=fixture["snapshot"], skill_id=CHILD_SKILL_ID,
                     steps=CHILD_STEPS, applicability=CHILD_APPLICABILITY)
    if not added.get("applied"):
        return {"check": "a_child_card_coexists_with_its_parent",
                "passed": False, "why": added.get("why"), "add": added}
    after = capability_cards(added["snapshot"])
    parent_after = after.get(K0_SKILL_ID)
    findings = {
        "the_parent_was_there_to_begin_with": parent_before is not None,
        "the_child_is_now_present": CHILD_SKILL_ID in after,
        "the_parent_is_still_present": parent_after is not None,
        "the_parents_body_is_unchanged": (
            parent_before is not None and parent_after is not None
            and parent_before["body"] == parent_after["body"]),
        "the_parents_scope_and_guards_are_unchanged": (
            parent_before is not None and parent_after is not None
            and parent_before["serving_scope"] == parent_after["serving_scope"]
            and parent_before["risk_guards"] == parent_after["risk_guards"]),
        "they_are_two_cards_not_one": len(after) == len(before) + 1,
        "no_other_card_changed": all(
            before[key] == after[key] for key in before if key in after),
        "the_child_still_owes_the_current_support": bool(
            after.get(CHILD_SKILL_ID, {}).get("requires_target_support")),
    }
    return {"check": "a_child_card_coexists_with_its_parent",
            "passed": all(findings.values()), "findings": findings,
            "cards_before": sorted(before), "cards_after": sorted(after),
            "parent_body": (parent_after or {}).get("body"),
            "operation": "EditOperation.ADD with surface_precondition ABSENT"}


def revoking_a_child_leaves_the_parent() -> dict[str, Any]:
    """Behaviour 2: the child's failure takes the child and nothing else.

    This is the exact cell DEV-AUTO-3 lost.  There the failing program *was*
    the card, so the revocation that followed a failed delayed gate deleted the
    ancestor with it; here the two are separate entries and the same removal
    reaches only one file.
    """
    fixture = _fixture("revoke")
    added = add_card(controller=fixture["controller"], store=fixture["store"],
                     snapshot=fixture["snapshot"], skill_id=CHILD_SKILL_ID,
                     steps=CHILD_STEPS, applicability=CHILD_APPLICABILITY)
    if not added.get("applied"):
        return {"check": "revoking_a_child_leaves_the_parent",
                "passed": False, "why": added.get("why")}
    with_child = capability_cards(added["snapshot"])
    result = remove_cards(store=fixture["store"], snapshot=added["snapshot"],
                          skill_ids=[CHILD_SKILL_ID])
    after = capability_cards(result["snapshot"])
    parent_before = with_child.get(K0_SKILL_ID)
    parent_after = after.get(K0_SKILL_ID)
    original = capability_cards(fixture["snapshot"])
    findings = {
        "the_child_was_present_before_the_revocation":
            CHILD_SKILL_ID in with_child,
        "the_child_is_gone": CHILD_SKILL_ID not in after,
        "exactly_one_entry_was_unlinked": len(result["files_unlinked"]) == 1,
        "the_parent_survived": parent_after is not None,
        "the_parents_body_is_byte_identical": (
            parent_before is not None and parent_after is not None
            and parent_before["body"] == parent_after["body"]),
        "the_parent_is_what_it_was_before_the_child_existed": (
            original.get(K0_SKILL_ID) == parent_after),
        "every_surviving_card_is_unchanged": all(
            with_child[key] == after[key] for key in after),
        "the_library_is_back_to_its_pre_child_membership":
            sorted(after) == sorted(original),
    }
    return {"check": "revoking_a_child_leaves_the_parent",
            "passed": all(findings.values()), "findings": findings,
            "removed": result["removed"],
            "files_unlinked": result["files_unlinked"],
            "cards_after": sorted(after)}


class _Episode:
    """The two fields ``_fast_winner_skill_id`` reads, and nothing else."""

    def __init__(self, workflow_signature: str) -> None:
        self.workflow_signature = workflow_signature
        self.task_consumer_key = "forecast|ridge|sMASE"


def a_second_card_for_the_same_program_is_refused() -> dict[str, Any]:
    """The refusal that is already in force, exercised rather than asserted.

    Two independent facts make it hold, and both are checked against the real
    code: the minted id is a function of the task scope and the winner's
    workflow signature, so the same program always lands on the same id; and
    the ADD carries ``surface_precondition={"kind": "ABSENT"}``, so an id that
    is already taken is refused before anything is written.
    """
    from SelfEvolvingHarnessTS.methods.ttha.method import (  # noqa: PLC0415
        _fast_winner_skill_id,
    )
    fixture = _fixture("duplicate")
    snapshot = fixture["snapshot"]
    cards = capability_cards(snapshot)
    parent = cards[K0_SKILL_ID]

    # 1. the id the mint would choose for the program the parent already holds
    ancestor_id = _fast_winner_skill_id(_Episode("outlier_mad"))
    other_id = _fast_winner_skill_id(_Episode("winsorize"))

    # 2. the ADD really is refused on a taken id, through the real controller
    first = add_card(controller=fixture["controller"], store=fixture["store"],
                     snapshot=snapshot, skill_id=CHILD_SKILL_ID,
                     steps=CHILD_STEPS, applicability=CHILD_APPLICABILITY)
    second = add_card(controller=fixture["controller"], store=fixture["store"],
                      snapshot=first["snapshot"], skill_id=CHILD_SKILL_ID,
                      steps=CHILD_STEPS, applicability=CHILD_APPLICABILITY)

    # 3. and the reading this package uses to report on a finished library
    same = duplicate_capability_card(
        snapshot, body=parent["body"],
        applicability=parent["observable_applicability"],
        serving_scope=parent["serving_scope"])
    different = duplicate_capability_card(
        snapshot, body="Frozen program steps: "
        + json.dumps([{"op": "winsorize", "params": {}}]),
        applicability=parent["observable_applicability"],
        serving_scope=parent["serving_scope"])
    findings = {
        "the_program_the_parent_holds_maps_onto_the_parents_id":
            ancestor_id == K0_SKILL_ID,
        "a_different_program_maps_elsewhere": other_id != K0_SKILL_ID,
        "the_first_add_succeeded": bool(first.get("applied")),
        "the_second_add_for_the_same_id_was_refused":
            not second.get("applied"),
        "and_it_was_refused_by_the_precondition_not_by_a_crash":
            "ABSENT" in str(second.get("why") or "").upper()
            or "PRECONDITION" in str(second.get("why") or "").upper()
            or "EXIST" in str(second.get("why") or "").upper(),
        "the_reading_recognises_a_duplicate": same == K0_SKILL_ID,
        "and_does_not_invent_one": different is None,
    }
    return {"check": "a_second_card_for_the_same_program_is_refused",
            "passed": all(findings.values()), "findings": findings,
            "id_for_the_ancestors_program": ancestor_id,
            "refusal": second.get("why"),
            "no_core_file_was_edited": (
                "method_contract is a harness dependency sha; editing "
                "methods/ttha/method.py would change every runtime_bundle_sha "
                "and invalidate the locked h0")}


CHECKS = (a_child_card_coexists_with_its_parent,
          revoking_a_child_leaves_the_parent,
          a_second_card_for_the_same_program_is_refused)


def run() -> dict[str, Any]:
    rows = []
    for check in CHECKS:
        try:
            rows.append(check())
        except Exception as exc:  # noqa: BLE001 - a fault is a failed check
            rows.append({"check": check.__name__, "passed": False,
                         "fault": "%s: %s" % (type(exc).__name__,
                                              str(exc)[:400])})
    return {"stage": "DEV_KNOW1_RETENTION_CHECKS",
            "passed": all(row["passed"] for row in rows),
            "llm_calls": 0, "consumer_fits": 0,
            "what_is_checked": ("only the two behaviours the work order names, "
                                "plus the duplicate guard the repair adds"),
            "checks": rows}


def main(argv: Sequence[str] | None = None) -> int:
    result = run()
    print("retention checks: %s" % ("PASS" if result["passed"] else "FAIL"))
    for row in result["checks"]:
        print("   %s %s" % ("ok  " if row["passed"] else "FAIL", row["check"]))
        if not row["passed"]:
            print("       %s" % json.dumps(
                {k: v for k, v in row.items() if k not in ("check", "passed")},
                ensure_ascii=False, default=str)[:900])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
