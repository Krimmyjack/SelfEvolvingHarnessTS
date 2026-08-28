"""SA-1 Part 1: the three write-backs that make a Skill an updatable hypothesis.

Until now a card was compiled once and frozen.  The SA-0 audit walked every
branch the production path can reach and found the same answer in all of them:
a conversion mints a new Target-local card and never touches the source, a
Support refusal is invisible to the card, and the one delayed-refusal writer
that could reach it is filtered to locally minted ``fast_winner_*`` ids.  So
"Skill is a hypothesis the evidence keeps revising" had no write path at all.

Three rules, all structured, none of them free text:

* **R1 positive** -- a supplied candidate cleared both gates: append one row to
  the card's own evidence ledger.  No tier change and no wider Scope; a
  positive earned under the card is Harness-conditioned and buys nothing.
* **R2 conflict** -- the card's Scope admitted a unit and the unit's Target
  refused the candidate: compile an exclusion from that unit's own binned
  Pattern view and narrow ``observable_applicability`` to ``all(old,
  not(exclusion))``.
* **R3 negative/harm** -- same narrowing, plus a structured demotion note.

Two constraints do most of the safety work here.

*Only contracted axes may appear in an exclusion.*  The applicability
evaluator is three-valued: a leaf naming a feature the current unit does not
carry evaluates to ``None``, an ``all`` containing ``None`` is ``None``, and
``evaluate_applicability`` reports a match only on ``True``.  An exclusion
written on an axis that can go missing therefore silently withholds the card
from units it was never meant to exclude.  The public extractor emits a fixed
mapping -- every key on every call (``runtime/public_features.py:309-331``) --
and that key set, intersected with what the edit schema can carry, is the set
this module will write on.  (SA-0 Q8.)

*Only a unit that actually refused may compile an exclusion.*  Narrowing is
autonomous because it can only ever supply less; predicting refusals is not,
because a wrong prediction costs a domain nobody measured.  What is lost by
narrowing too far can be bought back at the ladder price -- one fresh unguided
positive in the excluded domain -- and what is lost by excluding a domain
pre-emptively cannot be, because the card is never there to be measured.
(SA-0 Q12.)
"""
from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The route table pairs exactly one confirmed cause with each of the two
# surfaces below, and this book changes neither the table nor the router.
RISK_GUARD_CAUSE = "RISK_GAP"
# `.observable_applicability` carries ``target_class: "applicability"``, and
# ``RETRIEVAL_MISS`` is the only cause the frozen table pairs with that class
# (``evaluation/minipipe/feedback/fault_routes.json:14``).  Its *name* is about
# the widening direction -- "should have been retrieved and was not" -- and no
# cause code in the table means "this Scope reaches too far".  SA-1 uses it as
# the authorization token for a narrowing PATCH and records the mismatch as the
# open Q1 residue rather than minting a code, which would be a new platform.
APPLICABILITY_CAUSE = "RETRIEVAL_MISS"

EVIDENCE_LEDGER_KEY = "evidence_ledger"
REVISION_LOG_KEY = "revision_log"
DEMOTION_KEY = "demotions"


def contracted_axes(legal_features: Sequence[str] | None = None
                    ) -> frozenset[str]:
    """Axes an exclusion leaf may name.

    Computed, not listed: the public extractor is run once on a constant
    series and its mapping keys are the axes that are always present.  If the
    extractor ever stops emitting one of them, this set shrinks with it
    instead of going stale.
    """
    import numpy as np

    from SelfEvolvingHarnessTS.runtime.public_features import (
        extract_public_features,
    )

    probe = np.arange(128, dtype=np.float64)
    emitted = set(extract_public_features(
        probe, task_kind="classification").mapping)
    emitted.discard("task_kind")
    if legal_features is not None:
        emitted &= {str(name) for name in legal_features}
    return frozenset(emitted)


def _binned(view: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in dict(view or {}).items()}


def compile_exclusion(
    *,
    refusing_view: Mapping[str, Any],
    source_views: Sequence[Mapping[str, Any]],
    axes: frozenset[str],
) -> dict[str, Any]:
    """The refusing unit's distinguishing values, as an exclusion clause.

    An axis qualifies when the refusing unit is present on it, every source
    the card was earned on is present on it, and every source disagrees with
    the refusing unit there.  The clause is the conjunction of those values,
    so it excludes the refusing unit and anything identical to it on all of
    them, and cannot exclude a unit that differs on even one -- which is what
    keeps a narrowing bounded rather than a guess about neighbours.

    Returns the leaves and the reason when there are none; an empty result is
    a result, not a failure, and the caller must not widen the axis set to
    manufacture one.
    """
    refusing = _binned(refusing_view)
    sources = [_binned(view) for view in source_views]
    leaves: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for axis in sorted(axes):
        if axis not in refusing:
            skipped[axis] = "absent_from_the_refusing_unit"
            continue
        if not sources or any(axis not in view for view in sources):
            skipped[axis] = "absent_from_a_source_the_card_was_earned_on"
            continue
        if all(view[axis] != refusing[axis] for view in sources):
            leaves.append({"feature": axis, "op": "==",
                           "value": refusing[axis]})
        else:
            skipped[axis] = "the_card_was_earned_at_this_same_value"
    return {
        "leaves": leaves,
        "skipped": skipped,
        "empty_because": (
            None if leaves
            else "the refusing unit agrees with the card's own evidence on "
                 "every contracted axis, so nothing distinguishes it and no "
                 "exclusion may be written"),
    }


def narrow_applicability(old_ast: Mapping[str, Any],
                         leaves: Sequence[Mapping[str, Any]]
                         ) -> dict[str, Any]:
    """``all(old, not(all(leaves)))`` -- monotone, and the old AST is intact.

    Nesting rather than rewriting is deliberate: the previous Scope stays
    readable inside the new one, so what the card claimed before a refusal
    and what the refusal removed are both auditable off the entry itself.
    """
    if not leaves:
        raise ValueError("a narrowing with no exclusion leaves is a no-op")
    return {"all": [dict(old_ast),
                    {"not": {"all": [dict(leaf) for leaf in leaves]}}]}


def append_evidence_row(guards: Mapping[str, Any],
                        row: Mapping[str, Any]) -> dict[str, Any]:
    """R1.  Append-only: the ledger is a list and nothing rewrites a row.

    The in-memory Episode ledger cannot provide this -- the delayed update
    replaces an Episode in its slot, so the Support-only reading of it stops
    existing (SA-0 Q5).  The card carries its own rows instead.
    """
    updated = dict(guards or {})
    ledger = list(updated.get(EVIDENCE_LEDGER_KEY) or [])
    ledger.append(dict(row))
    updated[EVIDENCE_LEDGER_KEY] = ledger
    return updated


def append_demotion(guards: Mapping[str, Any],
                    note: Mapping[str, Any]) -> dict[str, Any]:
    """R3's structured note.  Fields only -- a Fast reader that has to parse a
    sentence to learn what was demoted is the C40 lesson."""
    updated = dict(guards or {})
    notes = list(updated.get(DEMOTION_KEY) or [])
    notes.append(dict(note))
    updated[DEMOTION_KEY] = notes
    return updated


def append_revision_log(guards: Mapping[str, Any],
                        entry: Mapping[str, Any]) -> dict[str, Any]:
    """Every revision writes one row about itself: which rule, which unit's
    reading triggered it, which surface moved, and the content sha before."""
    updated = dict(guards or {})
    log = list(updated.get(REVISION_LOG_KEY) or [])
    log.append(dict(entry))
    updated[REVISION_LOG_KEY] = log
    return updated


# --------------------------------------------------------------------------- #
# the write channel -- the same frozen EditController path L1 installed with
# --------------------------------------------------------------------------- #
def _controller(store_root: Path, base: Any, tag: str) -> tuple[Any, Any]:
    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
        EditController, FaultRouter, SurfaceRegistry,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore

    root = Path(store_root) / tag
    if root.exists():
        shutil.rmtree(root)
    store = SnapshotStore(root / "snapshots")
    store.materialize(base)
    store.set_active(base.runtime_bundle_sha)
    return store, EditController(store, surfaces=SurfaceRegistry(),
                                 router=FaultRouter())


def _skill_by_id(snapshot: Any, skill_id: str) -> Any:
    return next((skill for skill in snapshot.skills
                 if str(skill.skill_id) == str(skill_id)), None)


def patch_card(
    base: Any,
    *,
    skill_id: str,
    store_root: Path,
    tag: str,
    risk_guards: Mapping[str, Any] | None = None,
    observable_applicability: Mapping[str, Any] | None = None,
    predicted_data_effect: Sequence[str] = ("skill_revised",),
) -> dict[str, Any]:
    """Apply one or both card PATCHes and return the new snapshot plus receipts.

    SHA precondition on every edit, one manifest per surface, and the fork is
    materialized content-addressed by the store, so the parent bytes stay
    recoverable and ``set_active(old_sha)`` is a rollback.
    """
    from SelfEvolvingHarnessTS.contracts.harness import (
        EditManifest, EditOperation,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (
        _resolve_apply_manifest,
    )

    from . import source_skill as ss

    store, controller = _controller(Path(store_root), base, tag)
    snapshot = base
    receipts: list[dict[str, Any]] = []
    plan = [
        ("risk_guards", risk_guards, RISK_GUARD_CAUSE),
        ("observable_applicability", observable_applicability,
         APPLICABILITY_CAUSE),
    ]
    for field, value, cause in plan:
        if value is None:
            continue
        surface = "skill_library.entries/%s.%s" % (skill_id, field)
        parent = store.materialize(snapshot)
        before = _skill_by_id(snapshot, skill_id)
        manifest = EditManifest(
            edit_id="sa1_%s_%s_%d" % (field, skill_id, len(receipts) + 1),
            base_harness_sha=snapshot.harness_content_sha,
            target_pattern_id="sa1-skill-revision",
            target_surface_id=surface,
            operation=EditOperation.PATCH,
            surface_precondition={
                "kind": "SHA",
                "sha": controller.surface_precondition_sha(parent, surface),
            },
            dependency_precondition_shas={},
            minimal_patch={"value": dict(value)},
            new_value=None,
            observable_applicability=None,
            # The closed M0 predicate vocabulary
            # (``methods/ttha/schemas/behavior_predicate_v1.json``) has one
            # member that states what a narrowing does and it is the true
            # claim: outside the excluded region the effective view does not
            # move.  An evidence append does not move it anywhere.
            predicted_agent_behavior_change=(
                "effective_view_unchanged_out_of_scope",),
            predicted_data_effect=tuple(predicted_data_effect),
            automatically_selected_risk_cases=(),
            falsification_condition=("card_still_supplied_on_the_refusing_unit"
                                     if field == "observable_applicability"
                                     else "evidence_row_absent",),
            patch_id=None,
        )
        receipt = controller.apply_to_fork(
            parent, _resolve_apply_manifest(manifest, snapshot),
            confirmed_cause=cause)
        snapshot = compile_snapshot(receipt.candidate_root, verify_lock=False)
        store.set_active(snapshot.runtime_bundle_sha)
        receipts.append({
            "surface": surface,
            "confirmed_cause": cause,
            "parent_runtime_bundle_sha": receipt.parent_runtime_bundle_sha,
            "candidate_runtime_bundle_sha": (
                receipt.candidate_runtime_bundle_sha),
            "card_sha_before": ss.skill_content_sha(before),
            "card_sha_after": ss.skill_content_sha(
                _skill_by_id(snapshot, skill_id)),
        })
    return {"snapshot": snapshot, "store": store, "receipts": receipts,
            "card_sha": ss.skill_content_sha(_skill_by_id(snapshot, skill_id))}
