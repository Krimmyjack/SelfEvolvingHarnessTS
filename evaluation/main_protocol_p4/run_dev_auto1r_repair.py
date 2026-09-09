"""DEV-AUTO-1R: one comparator, and three acceptance frames read offline.

What this package is
--------------------
A repair pass over DEV-AUTO-1, not a new experiment.  It re-reads that run's
own receipts (``dev_auto1_skill_revision__run2.json``,
``dev_auto1_blocked_candidates__run2.json``) and the shared prediction store,
fixes the comparison it made, and asks what three different acceptance rules
would have done with the twenty-five proposals that actually occurred.

The three things it repairs
---------------------------
1. **The denominator.**  DEV-AUTO-1 scored a candidate over the cells the
   candidate was readable on and the parent over the cells the parent was
   readable on, and called the difference a comparison.  At position 16 that is
   a mean over 7 cells against a mean over 15.  Everything here goes through
   ``dev_auto1_revision.compare_on_a_common_denominator``, which names one unit
   set for both sides and keeps READ / RAW_FALLBACK / UNKNOWN apart.

2. **What the parent comparison is a comparison *with*.**  Replaying the
   parent *program* diagnoses that program.  It is not the policy the arm
   actually executed -- in this course Fast deployed a program in 3 of 32 cells
   and served raw everywhere else.  Both columns are computed and they are
   never mixed.

3. **Cross-unit worst versus per-unit risk.**  "Not worse on the worst cell"
   and "not worse on any cell" are different claims.  Both are reported.

The parent's per-cell readings
------------------------------
DEV-AUTO-1 stored only the parent's *summary*, so a per-unit comparison needs
the parent's cells back.  They are reconstructed from the shared prediction
store's ANCESTOR entries with ``m_r0k._reading_from_entry`` -- no fits -- and
the reconstruction is checked bit for bit against every parent summary
DEV-AUTO-1 recorded before it is used.  A mismatch stops the package.

Cost
----
0 LLM calls.  Cached readings first; new fits only for the forward-validation
face of a program the store has never held, capped at ``MAX_NEW_FITS`` and
billed to this package alone.  No Consumer, DSL, risk threshold, risk
denominator or Scope grid is touched, and no historical receipt is overwritten.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_m_r0k_scope_workflow_development as m_r0k

from evaluation.main_protocol_p4 import p4b_contract as bounded
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

ART = base.ROOT / "artifacts" / "main_protocol"
OUT = ART / "dev_auto1r_repair.json"

MAX_NEW_FITS = 100
#: This package's own fit ledger, so a second run of the script does not get a
#: fresh 100 by finding the first run's entries already in the store.  The
#: ceiling is the package's, not the process's.
FIT_LEDGER = base.ROOT / "_scratch" / "dev_auto1r_fit_ledger.json"
MAX_WALL_SECONDS = 2 * 3600
FITS_PER_ENTRY = 2

MATERIAL = float(contract.RISK["material"])
SUPPORT_FACE = "support_face"
DELAYED_FACE = "delayed_face"


class Ceiling(RuntimeError):
    """The package's own fit ceiling, refused before the spend."""


# ---------------------------------------------------------------------------
# the course, and the parent's cells
# ---------------------------------------------------------------------------

def _course() -> dict[str, Any]:
    doc = base.load(R.ORDERING)
    population, excluded = R._population(doc)
    by_position = {row["position"]: row for row in population}
    key_to_position = {(row["unit"]["block"], row["unit"]["origin"]):
                       row["position"] for row in population}
    assert len(key_to_position) == len(population), "unit keys are not unique"
    return {"population": population, "excluded": excluded,
            "by_position": by_position, "key_to_position": key_to_position,
            "after": [row for row in population
                      if row["side"] == "after_the_boundary"]}


def _ancestor_cells(course: Mapping[str, Any], store: Mapping[str, Any],
                    face: str) -> dict[int, dict[str, Any]]:
    """The parent program's reading on every unit, from the store.  No fits."""
    out: dict[int, dict[str, Any]] = {}
    for row in course["population"]:
        ctx = m_r0k.opp._unit_ctx(row["unit"])
        origin = m_r0k._faces_for(ctx)[face]
        entry = store.get(m_r0k._entry_key("ANCESTOR", row["position"], face))
        if entry is None:
            out[row["position"]] = {"status": "NO_STORED_READING"}
            continue
        resolved = frozenset(ctx.resolve(R.ANCESTOR_SCOPE, origin))
        out[row["position"]] = m_r0k._reading_from_entry(entry, resolved)
    return out


def _check_the_parent_reconstruction(report: Mapping[str, Any],
                                     course: Mapping[str, Any],
                                     ancestor: Mapping[int, Mapping[str, Any]],
                                     ) -> dict[str, Any]:
    """Every parent summary DEV-AUTO-1 recorded, rebuilt from the store."""
    checked, mismatched = 0, []
    for step in report["revision_steps"]:
        positions = _step_positions(step, course)
        if positions is None:
            continue
        got = [ancestor[p] for p in positions]
        if any(row.get("status") != rev.READ for row in got):
            mismatched.append({"position": step["position"],
                               "why": "a parent cell is not READ in the store"})
            continue
        mine = {
            "mean_aggregate_gain_over_the_served_population":
                round(statistics.fmean([r["aggregate_gain"] for r in got]), 6),
            "worst_single_series_harm":
                round(max(r["max_single_series_harm"] for r in got), 6),
            "worst_harmed_fraction":
                round(max(r["harmed_fraction"] for r in got), 6),
        }
        theirs = step["parent_policy"]["screen"]
        if any(mine[k] != theirs[k] for k in mine):
            mismatched.append({"position": step["position"],
                               "reconstructed": mine,
                               "recorded": {k: theirs[k] for k in mine}})
        else:
            checked += 1
    return {"parent_steps_reproduced_bit_for_bit": checked,
            "mismatches": mismatched,
            "source": ("the shared prediction store's ANCESTOR entries, "
                       "re-masked by m_r0k._reading_from_entry under the "
                       "ancestor serving scope"),
            "fits_spent": 0}


def _step_positions(step: Mapping[str, Any],
                    course: Mapping[str, Any]) -> list[int] | None:
    """The processed cells this step's screen actually ran over.

    Read off the candidate's own ``per_cell`` list, which enumerates every cell
    including the ones it could not read.  A step with no workflow evaluation
    leaves no such list and returns None -- its unit identities are not
    recoverable from the receipt, and guessing them is exactly the kind of fill
    this package exists to stop.
    """
    for evaluation in step.get("evaluations") or ():
        cells = (evaluation.get("screen") or {}).get("per_cell") or ()
        if cells:
            return [course["key_to_position"][(c["unit"]["block"],
                                               c["unit"]["origin"])]
                    for c in cells]
    return None


def _unit_list(course: Mapping[str, Any],
               positions: Sequence[int]) -> list[dict[str, Any]]:
    return [{"position": p, "unit": course["by_position"][p]["unit"]}
            for p in positions]


def _rows_by_key(course: Mapping[str, Any], positions: Sequence[int],
                 reading_of) -> dict[str, dict[str, Any]]:
    return {"pos%d" % p: rev.classify_reading(reading_of(p)) for p in positions}


# ---------------------------------------------------------------------------
# 1. the twenty-five proposals, recomputed on one denominator
# ---------------------------------------------------------------------------

def _executed_policy_cells(report: Mapping[str, Any], course: Mapping[str, Any],
                           arm: str, face: str) -> dict[int, dict[str, Any]]:
    """What the arm actually served, unit by unit, from this course's cells.

    A unit the course never executed -- the five pre-boundary units are entry
    state, not cells this run produced -- has no reading here and stays
    UNKNOWN.  It is not the same thing as a unit that delivered nothing.
    """
    key = "support_reading" if face == SUPPORT_FACE else "delayed_reading"
    out: dict[int, dict[str, Any]] = {}
    for cell in report["cells"]:
        if cell["arm"] != arm:
            continue
        reading = cell.get(key)
        out[int(cell["position"])] = (
            {**reading, "status": rev.READ} if reading
            else {"status": "NOT_EXECUTED_IN_THIS_COURSE"})
    return out


def recompute_the_proposals(report: Mapping[str, Any], course: Mapping[str, Any],
                            ancestor: Mapping[int, Mapping[str, Any]],
                            executed: Mapping[int, Mapping[str, Any]],
                            ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in report["revision_steps"]:
        positions = _step_positions(step, course)
        for index, evaluation in enumerate(step.get("evaluations") or ()):
            row: dict[str, Any] = {
                "position": step["position"],
                "index": index,
                "kind": evaluation.get("kind"),
                "program": evaluation.get("program_replayed"),
                "scope_clause": (evaluation.get("scope_clause")
                                 or (evaluation.get("calibration") or {}).get(
                                     "feature")),
                "recorded_outcome": evaluation.get("outcome"),
                "recorded_failed_lines": evaluation.get("failed_lines"),
                "legal": bool((evaluation.get("legality") or {}).get("legal")),
            }
            cells = (evaluation.get("screen") or {}).get("per_cell") or ()
            if not cells or positions is None:
                row["comparison"] = None
                row["why_no_comparison"] = (
                    "this proposal produced no replay: %s"
                    % evaluation.get("outcome"))
                rows.append(row)
                continue

            by_position = {course["key_to_position"][(c["unit"]["block"],
                                                      c["unit"]["origin"])]: c
                           for c in cells}
            declared = sorted(by_position)
            units = _unit_list(course, declared)
            candidate = _rows_by_key(course, declared, by_position.get)

            row["against_the_parent_program"] = \
                rev.compare_on_a_common_denominator(
                    units=units, candidate=candidate,
                    reference=_rows_by_key(course, declared, ancestor.get),
                    face=SUPPORT_FACE,
                    candidate_label=str(row["program"]),
                    reference_label="the parent program on the Skill card",
                    comparison_kind=(
                        "parent PROGRAM replay -- a diagnosis of the program, "
                        "not of the policy the arm executed"))

            executed_positions = [p for p in declared if p in executed
                                  and executed[p].get("status") == rev.READ]
            row["against_the_executed_policy"] = \
                rev.compare_on_a_common_denominator(
                    units=_unit_list(course, executed_positions),
                    candidate=_rows_by_key(course, executed_positions,
                                           by_position.get),
                    reference=_rows_by_key(course, executed_positions,
                                           executed.get),
                    face=SUPPORT_FACE,
                    candidate_label=str(row["program"]),
                    reference_label="what the revising arm actually served",
                    comparison_kind=(
                        "EXECUTED policy -- only the units this course ran; "
                        "the pre-boundary units are entry state and carry no "
                        "reading here"))
            row["executed_policy_units_available"] = len(executed_positions)
            row["executed_policy_units_missing"] = [
                p for p in declared if p not in executed_positions]
            rows.append(row)
    return rows


def restate_the_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The "9 beat the parent, 5 no worse on either risk line" line, redone.

    DEV-AUTO-1 derived those from ``relative_to_the_parent_screen``, whose two
    means were taken over different cell sets.  They are recomputed here from
    the common-denominator comparison and the derived booleans are not reused.
    """
    old_beats, old_kept = 0, 0
    for row in rows:
        pass
    beats, kept_summary, kept_every_unit, undetermined = [], [], [], []
    for row in rows:
        comparison = row.get("against_the_parent_program")
        if not comparison:
            continue
        tag = "u%s#%d %s" % (row["position"], row["index"], row["program"])
        if not comparison["verdict_is_complete"]:
            undetermined.append(tag)
            continue
        gain = comparison["mean_aggregate_gain"]
        worst = comparison["cross_unit_worst"]
        per_unit = comparison["per_unit_risk"][
            "units_where_the_candidate_is_worse"]
        if gain["candidate_beats_the_reference_by_material"]:
            beats.append(tag)
            if not any(worst[name]["candidate_worse"]
                       for name in ("single_series_harm", "harmed_fraction")):
                kept_summary.append(tag)
                if not any(per_unit[name]
                           for name in ("single_series_harm",
                                        "harmed_fraction")):
                    kept_every_unit.append(tag)
    return {
        "what_DEV_AUTO_1_reported": {
            "beat_the_parent_by_material": 9,
            "and_no_worse_on_either_risk_line": 5,
            "computed_how": ("each side's mean over its own readable cells, "
                             "and a boolean derived from those two means"),
        },
        "recomputed_on_one_denominator": {
            "beat_the_parent_by_material": len(beats),
            "and_no_worse_on_either_summarised_risk_line": len(kept_summary),
            "and_no_worse_on_any_single_unit": len(kept_every_unit),
            "undetermined": len(undetermined),
        },
        "which": {"beats_the_parent": beats,
                  "no_worse_in_the_summary": kept_summary,
                  "no_worse_on_every_unit": kept_every_unit,
                  "undetermined": undetermined},
        "note": ("'no worse on the cross-unit worst' and 'no worse on every "
                 "unit' are different counts and are reported as two"),
    }


# ---------------------------------------------------------------------------
# 2. the position-23 Support gap
# ---------------------------------------------------------------------------

def close_the_pos23_gap(course: Mapping[str, Any], store: dict[str, Any],
                        cache: runner.ReplayPredictionCache,
                        spend) -> dict[str, Any]:
    """Fill or name the two Support cells the audit scored as if refused.

    ``dev_auto1_blocked_candidates__run2.json`` carries
    ``{"position": 23, "origin": null, "status": "NO_STORED_READING"}`` for both
    candidates on the Support face, and then scored that face at 0.0 alongside
    the genuinely-refused ones.  A missing reading is not a refusal.
    """
    out: list[dict[str, Any]] = []
    for label, steps in _blocked_program_steps().items():
        if label not in ("outlier_iqr({})>winsorize({})",
                         "outlier_iqr({})>hampel_filter({})"):
            continue
        row = course["by_position"][23]
        ctx = m_r0k.opp._unit_ctx(row["unit"])
        origin = m_r0k._faces_for(ctx)[SUPPORT_FACE]
        record: dict[str, Any] = {"program": label, "position": 23,
                                  "face": SUPPORT_FACE, "origin": origin}
        try:
            entry = _entry(label, 23, SUPPORT_FACE, ctx, origin, tuple(steps),
                           store, cache, spend)
            resolved = frozenset(ctx.resolve(R.ANCESTOR_SCOPE, origin))
            reading = m_r0k._reading_from_entry(entry, resolved)
            record["reading"] = reading
            record["state"] = rev.classify_reading(reading)["state"]
        except Ceiling as exc:
            record.update({"state": rev.UNKNOWN, "status": "REFUSED_AT_THE_CEILING",
                           "why": str(exc)})
        except Exception as exc:  # noqa: BLE001 - the fault is the reading
            record.update({"state": rev.UNKNOWN,
                           "status": type(exc).__name__,
                           "why": str(exc)[:200]})
        out.append(record)
    determined = [r for r in out if r.get("state") != rev.UNKNOWN]
    return {
        "what_the_audit_did": ("scored position 23 Support at 0.0 for both "
                               "candidates under the raw-fallback rule, having "
                               "recorded NO_STORED_READING for it"),
        "why_that_is_wrong": ("the raw-fallback rule is for a pair the harness "
                              "confirmed it cannot execute; this cell was "
                              "never determined"),
        "attempts": out,
        "now_determined": len(determined),
        "still_unknown": [r["program"] for r in out
                          if r.get("state") == rev.UNKNOWN],
    }


def _blocked_program_steps() -> dict[str, tuple]:
    report = json.loads((ART / "dev_auto1_skill_revision__run2.json").read_text(
        encoding="utf-8"))
    out: dict[str, tuple] = {}
    for step in report["revision_steps"]:
        for evaluation in step.get("evaluations") or ():
            if evaluation.get("kind") != "workflow_program":
                continue
            label = evaluation.get("program_replayed")
            if label in out:
                continue
            out[label] = tuple(
                (row["op"], dict(row.get("params") or {}))
                for row in evaluation.get("steps_replayed") or ())
    return out


def _entry(label: str, position: int, face: str, ctx: Any, origin: int,
           steps: tuple, store: dict[str, Any],
           cache: runner.ReplayPredictionCache, spend) -> dict[str, Any]:
    """One stored entry, from the store if it is there and paid for if not."""
    key = m_r0k._entry_key(label, position, face)
    if key in store:
        return store[key]
    spend(FITS_PER_ENTRY)
    before = cache.physical_fits
    built = cache._build(ctx, origin, steps)
    store[key] = {
        "program": label, "position": position, "face": face, "origin": origin,
        "verifier_passed": bool(built.get("verifier_passed")),
        "eval_uids": list(built.get("eval_uids") or ()),
        "raw_per_view": [float(v) for v in (built.get("raw_per_view") or ())],
        "program_per_view": [float(v)
                             for v in (built.get("program_per_view") or ())],
        "degenerate_uids": list(built.get("degenerate_uids") or ()),
        "physical_fits": int(cache.physical_fits - before),
    }
    m_r0k._save_store(store)
    return store[key]


def recheck_the_audit(audit: Mapping[str, Any], course: Mapping[str, Any],
                      pos23: Mapping[str, Any]) -> dict[str, Any]:
    """The post-hoc audit's headline, through the same comparator.

    The audit scored every unreadable face at 0.0 under the raw-fallback rule.
    That is right for the faces the harness confirmed it cannot execute and
    wrong for position 23 on the Support face, which nobody ever read.  Here
    the two cases are separated, so the reader can see how much of the
    published difference rests on a cell that was never measured.
    """
    unknown_by_program = {row["program"]: row for row in pos23["attempts"]}
    out: dict[str, Any] = {}
    for label, rows in audit["per_unit"].items():
        if label == "ANCESTOR":
            continue
        ancestor_rows = {r["position"]: r for r in audit["per_unit"]["ANCESTOR"]}
        program = label[len("CAND_"):] if label.startswith("CAND_") else label
        note = unknown_by_program.get(program)
        for face in (SUPPORT_FACE, DELAYED_FACE):
            positions, candidate, reference = [], {}, {}
            for row in rows:
                position = int(row["position"])
                positions.append(position)
                mine = (row["faces"].get(face) or {}).get("reading")
                if mine is None:
                    mine = {"status": (note or {}).get("status")
                            or "NO_STORED_READING",
                            "why": (note or {}).get("why")}
                theirs = (ancestor_rows[position]["faces"].get(face)
                          or {}).get("reading")
                candidate["pos%d" % position] = rev.classify_reading(mine)
                reference["pos%d" % position] = rev.classify_reading(theirs)
            comparison = rev.compare_on_a_common_denominator(
                units=_unit_list(course, sorted(positions)),
                candidate=candidate, reference=reference, face=face,
                candidate_label=program,
                reference_label="the ancestor program",
                comparison_kind=("post-hoc audit over the sixteen units after "
                                 "the boundary, both sides on the same units"))
            published = ((audit["headline_on_the_16_units_after_the_boundary"]
                          .get(label) or {}).get(face) or {})
            out.setdefault(program, {})[face] = {
                "recomputed": comparison["mean_aggregate_gain"],
                "published_by_the_audit": {
                    "mean_aggregate_gain": published.get("mean_aggregate_gain"),
                    "vs_the_ancestor": published.get("vs_the_ancestor"),
                },
                "cell_states": comparison["candidate_cell_states"],
                "verdict_is_complete": comparison["verdict_is_complete"],
                "why_the_verdict_is_withheld":
                    comparison["why_the_verdict_is_withheld"],
                "cross_unit_worst": comparison["cross_unit_worst"],
                "per_unit_risk": comparison["per_unit_risk"][
                    "units_where_the_candidate_is_worse"],
            }
    return {
        "what_changed": ("position 23 on the Support face is UNKNOWN, not a "
                         "raw fallback: the audit had no reading for it and "
                         "scored it 0.0 anyway"),
        "consequence": ("with it excluded from both sides the Support-face "
                        "comparison is over 15 units, not 16, and the "
                        "published Support difference is not a complete "
                        "verdict"),
        "the_instrument_note": (
            "the ancestor IS readable at position 23 Support and these two "
            "programs are not -- FaceNotEvaluable, 'evaluation context reached "
            "scale floor', reproduced here at zero fits.  run_hec1 documents "
            "that class as 'a property of the data, not of a policy, so it "
            "hits every arm on that unit identically'; on this cell it does "
            "not, and that is an instrument observation this package records "
            "rather than acts on"),
        "per_program": out,
    }


# ---------------------------------------------------------------------------
# 3. three acceptance frames, and 4. forward validation
# ---------------------------------------------------------------------------

REFERENCE_DECLARATION = (
    "Frame B's reference is fixed before any number is computed: the policy "
    "that legally held deployment rights at that step -- the program on the "
    "Skill card under its then-current serving scope, which DEV-AUTO-1 "
    "recorded as parent_policy before it scored any candidate.  It is never "
    "swapped for raw, and never chosen after the fact for being the easier "
    "bar.  It is a PROGRAM reference: on most of these cells the arm actually "
    "served raw, and that column is reported separately rather than "
    "substituted here."
)


def frame_decisions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        comparison = row.get("against_the_parent_program")
        calibrated = row["kind"] != "scope_clause" or (
            row["recorded_outcome"] not in ("NO_FEASIBLE_THRESHOLD",
                                            "SLOW_CLAUSE_UNUSABLE",
                                            "NARROWING_REFUSED"))
        shared = {"legal": row["legal"], "calibrated": bool(calibrated)}

        # -- A: the historical absolute screen, exactly as it ran, with the
        #       utility half recomputed on the common denominator.
        screen_clean = row["recorded_outcome"] not in (
            "RISK_LINE_FAILED", "NOT_READABLE")
        beats = (comparison["mean_aggregate_gain"][
                     "candidate_beats_the_reference_by_material"]
                 if comparison and comparison["verdict_is_complete"] else None)
        a = {"enters_verification": bool(
                shared["legal"] and shared["calibrated"]
                and screen_clean and beats),
             "why": ([] if screen_clean else
                     ["eliminated_by_the_absolute_historical_screen"])
                    + ([] if shared["calibrated"] else ["no_legal_calibration"])
                    + ([] if beats else ["not_material_over_the_reference"])}

        # -- B: the reference bar, no absolute historical screen.
        b_verdict = (rev.frame_b_admits(comparison) if comparison
                     else {"admits": None,
                           "why": ["UNDETERMINED: no replay was produced"]})
        b = {"enters_verification": bool(shared["legal"] and shared["calibrated"]
                                         and b_verdict["admits"]),
             "why": ([] if shared["calibrated"] else ["no_legal_calibration"])
                    + list(b_verdict["why"]),
             "undetermined": b_verdict["admits"] is None}

        # -- C: legal and calibratable enters; replay orders, it does not veto.
        c = {"enters_verification": bool(shared["legal"] and shared["calibrated"]),
             "why": ([] if shared["calibrated"] else ["no_legal_calibration"]),
             "ordering_key": (comparison["mean_aggregate_gain"]["difference"]
                              if comparison
                              and comparison["verdict_is_complete"] else None)}

        out.append({**{k: row[k] for k in
                       ("position", "index", "kind", "program",
                        "recorded_outcome")},
                    "shared_checks": shared,
                    "A_absolute_historical_screen": a,
                    "B_relative_reference": b,
                    "C_forward_validation_first": c})
    return out


def _queue(decisions: Sequence[Mapping[str, Any]], frame: str,
           ) -> list[dict[str, Any]]:
    """Distinct programs a frame admits, each at the position it first got in."""
    first: dict[str, dict[str, Any]] = {}
    for row in decisions:
        if not row[frame]["enters_verification"]:
            continue
        if row["kind"] != "workflow_program":
            # A Scope revision is verified through the Draft ledger, not by
            # replaying a program on later units; none reached here anyway.
            continue
        label = row["program"]
        if label not in first or row["position"] < first[label]["position"]:
            first[label] = {"program": label, "position": row["position"],
                            "ordering_key": row["C_forward_validation_first"][
                                "ordering_key"]}
    return sorted(first.values(),
                  key=lambda r: (r["position"], r["program"]))


def forward_validate(queue: Sequence[Mapping[str, Any]],
                     course: Mapping[str, Any], store: dict[str, Any],
                     cache: runner.ReplayPredictionCache, spend,
                     steps_by_label: Mapping[str, tuple]) -> list[dict[str, Any]]:
    """Walk each candidate forward from where it was actually proposed.

    Course order only, no window picking.  Support uses the production
    admission rule -- the risk budget in ``admission_policy``, plus the
    material line the online loop applies before a candidate can win -- and no
    coverage floor is added to it.  delayed uses the existing authoritative
    gate.  Attempts stop at ``MAX_VERIFICATION_ATTEMPTS``; a pass is a pass at
    the first window that gives one, not the best window found by scanning.
    """
    results: list[dict[str, Any]] = []
    positions = [row["position"] for row in course["after"]]
    for item in queue:
        label = item["program"]
        steps = steps_by_label[label]
        later = [p for p in positions if p > item["position"]]
        record: dict[str, Any] = {
            "program": label, "proposed_at_position": item["position"],
            "later_units_available": len(later), "attempts": [],
        }
        if not later:
            record["outcome"] = "NO_LATER_UNIT"
            results.append(record)
            continue
        outcome = None
        record["windows_not_evaluated"] = []
        for position in later:
            if len(record["attempts"]) >= drafts.MAX_VERIFICATION_ATTEMPTS:
                outcome = "VERIFICATION_ATTEMPTS_EXHAUSTED"
                break
            row = course["by_position"][position]
            ctx = m_r0k.opp._unit_ctx(row["unit"])
            faces = m_r0k._faces_for(ctx)
            attempt: dict[str, Any] = {"position": position,
                                       "support_origin": faces[SUPPORT_FACE]}
            try:
                support = m_r0k._reading_from_entry(
                    _entry(label, position, SUPPORT_FACE, ctx,
                           faces[SUPPORT_FACE], steps, store, cache, spend),
                    frozenset(ctx.resolve(R.ANCESTOR_SCOPE,
                                          faces[SUPPORT_FACE])))
            except Ceiling as exc:
                attempt.update({"support": "NOT_EVALUATED",
                                "why": "REFUSED_AT_THE_CEILING: %s" % exc})
                record["attempts"].append(attempt)
                outcome = "NOT_EVALUATED"
                break
            except Exception as exc:  # noqa: BLE001 - recorded as a reading
                # An unmeasurable window is a gap in the instrument, not a
                # verification this candidate failed.  It is recorded by name
                # and does not consume one of the three attempts -- charging a
                # candidate for a face nobody could read would be the same
                # error as scoring that face zero.
                attempt.update({"support": "NOT_EVALUATED",
                                "why": "%s: %s" % (type(exc).__name__,
                                                   str(exc)[:160])})
                record["windows_not_evaluated"].append(attempt)
                continue
            state = rev.classify_reading(support)
            attempt["support_state"] = state["state"]
            attempt["support_status"] = state["status"]
            if state["state"] != rev.READ:
                # A confirmed refusal is a real answer: the policy serves raw
                # here and there is nothing to admit.  It consumes an attempt.
                attempt["support_admits"] = False
                attempt["support_reason"] = state["status"]
                record["attempts"].append(attempt)
                continue
            admits = _support_admits(support)
            attempt["support"] = {k: support[k] for k in
                                  ("treated", "served", "aggregate_gain",
                                   "harmed_fraction", "max_single_series_harm")}
            attempt.update(admits)
            if not admits["support_admits"]:
                record["attempts"].append(attempt)
                continue
            try:
                delayed = m_r0k._reading_from_entry(
                    _entry(label, position, DELAYED_FACE, ctx,
                           faces[DELAYED_FACE], steps, store, cache, spend),
                    frozenset(ctx.resolve(R.ANCESTOR_SCOPE,
                                          faces[DELAYED_FACE])))
            except Ceiling as exc:
                attempt.update({"delayed": "NOT_EVALUATED",
                                "why": "REFUSED_AT_THE_CEILING: %s" % exc})
                record["attempts"].append(attempt)
                outcome = "NOT_EVALUATED"
                break
            except Exception as exc:  # noqa: BLE001
                attempt.update({"delayed": "NOT_EVALUATED",
                                "why": "%s: %s" % (type(exc).__name__,
                                                   str(exc)[:160])})
                record["windows_not_evaluated"].append(attempt)
                continue
            dstate = rev.classify_reading(delayed)
            attempt["delayed_state"] = dstate["state"]
            if dstate["state"] != rev.READ:
                attempt["delayed_gate"] = {"passes": False,
                                           "failed_lines": [dstate["status"]]}
                record["attempts"].append(attempt)
                continue
            gate = runner.authoritative_gate(delayed)
            attempt["delayed"] = {k: delayed[k] for k in
                                  ("treated", "served", "aggregate_gain",
                                   "harmed_fraction", "max_single_series_harm")}
            attempt["delayed_gate"] = {"passes": gate["passes"],
                                       "failed_lines": gate["failed_lines"]}
            record["attempts"].append(attempt)
            if gate["passes"]:
                outcome = "VALIDATION_PASSED"
                record["passed_at_position"] = position
                break
        if outcome is None:
            outcome = ("VERIFICATION_ATTEMPTS_EXHAUSTED"
                       if len(record["attempts"])
                       >= drafts.MAX_VERIFICATION_ATTEMPTS
                       else "NO_WINDOW_PASSED")
        record["outcome"] = outcome
        record["attempts_used"] = len(record["attempts"])
        record["max_attempts"] = drafts.MAX_VERIFICATION_ATTEMPTS
        record["windows_skipped_as_not_evaluated"] = len(
            record["windows_not_evaluated"])
        record["promotion_still_requires"] = (
            "Fast deploying this program in the unit whose delayed gate "
            "passed; validation passing is not promotion")
        results.append(record)
    return results


def _support_admits(reading: Mapping[str, Any]) -> dict[str, Any]:
    """Production Support admission: the risk budget, plus the material line.

    ``admission_policy._budget_verdict`` is the harness's own rule -- harmed
    fraction and worst single-series harm against the released constants -- and
    it carries no coverage floor.  None is added here.
    """
    # The policy the P4 runners install (``run_hec1`` line 2722), not the
    # module default, which is the strict gate and would refuse everything.
    policy = bounded.BOUNDED_POLICY
    verdict = admission_policy._budget_verdict(
        aggregate=float(reading["aggregate_gain"]),
        series_count=int(reading["served"]),
        harmed_count=int(reading["harmed_series"]),
        worst_harm=float(reading["max_single_series_harm"]),
        policy=policy)
    material_ok = float(reading["aggregate_gain"]) >= MATERIAL
    return {
        "support_admits": bool(verdict.admitted and material_ok),
        "support_reason": (str(verdict.reason) if not verdict.admitted
                           else ("within_risk_budget" if material_ok
                                 else "aggregate_below_material")),
        "support_rule": ("admission_policy._budget_verdict (harmed_fraction "
                         "<= %.2f, single-series harm <= %.2f) and the "
                         "material line the online loop applies before a "
                         "candidate can win; no coverage floor"
                         % (policy.max_harmed_fraction,
                            policy.max_single_series_harm)),
    }


def what_stopped_fast(report: Mapping[str, Any]) -> dict[str, Any]:
    """Why 20 of 32 cells served identity: the rule, not a missed candidate.

    DEV-AUTO-1 read "probing gains up to +0.407011 and deploying in 3 of 32
    cells" as a selection failure.  It is not.  Every probe carries its own
    admission verdict and risk profile, and re-deciding those rows says three
    separate things:

    * the run was under ``admission_policy.DEFAULT`` -- the *strict* rule,
      which admits only a probe that harms no series at all.  No production
      runner uses it; they all install the released bounded policy;
    * the +0.407011 probe is refused by BOTH rules (harmed fraction 0.25
      against a line of 0.20), so it was never a good candidate the selector
      missed;
    * strict refuses with ``relation_not_positive``, which is not one of
      ``RISK_REFUSAL_REASONS``, so ``risk_refusal_count`` stayed 0 in all 32
      cells and no probe was ever routed as a risk problem.
    """
    from evaluation.main_protocol_p4 import p4b_contract as bounded
    from SelfEvolvingHarnessTS.methods.ttha import admission_policy as ap
    rows: list[dict[str, Any]] = []
    for cell in report["cells"]:
        for probe in cell.get("probes") or ():
            profile = probe.get("risk_profile") or {}
            verdict = probe.get("admission") or {}
            gain = probe.get("gain")
            if gain is None or profile.get("harmed_fraction") is None:
                continue
            served = int(probe.get("served") or 20)
            bounded_verdict = ap._budget_verdict(
                aggregate=float(gain), series_count=served,
                harmed_count=int(round(float(profile["harmed_fraction"])
                                       * served)),
                worst_harm=float(profile["max_single_series_harm"]),
                policy=bounded.BOUNDED_POLICY)
            rows.append({
                "arm": cell["arm"], "position": cell["position"],
                "program": rev.program_label(tuple(
                    (s["op"], dict(s.get("params") or {}))
                    for s in probe["program_steps"])),
                "gain": round(float(gain), 6),
                "harmed_fraction": float(profile["harmed_fraction"]),
                "max_single_series_harm": float(
                    profile["max_single_series_harm"]),
                "admitted_by_the_rule_that_ran": bool(verdict.get("admitted")),
                "reason_recorded": str(verdict.get("reason")),
                "would_be_admitted_by_the_released_rule": bool(
                    bounded_verdict.admitted and float(gain) >= MATERIAL),
                "released_rule_reason": str(bounded_verdict.reason),
            })
    changed = [row for row in rows
               if row["admitted_by_the_rule_that_ran"]
               != row["would_be_admitted_by_the_released_rule"]]
    best = max(rows, key=lambda r: r["gain"]) if rows else None
    return {
        "the_rule_that_ran": ap.DEFAULT.to_dict(),
        "the_released_rule_every_production_runner_installs":
            bounded.BOUNDED_POLICY.to_dict(),
        "probe_rows_with_a_risk_profile": len(rows),
        "admitted_by_the_rule_that_ran": sum(
            1 for r in rows if r["admitted_by_the_rule_that_ran"]),
        "would_be_admitted_by_the_released_rule": sum(
            1 for r in rows if r["would_be_admitted_by_the_released_rule"]),
        "rows_the_rule_change_would_flip": changed,
        "the_highest_gain_probe": best,
        "is_the_highest_gain_probe_a_missed_candidate": (
            None if best is None
            else bool(best["would_be_admitted_by_the_released_rule"])),
        "risk_refusals_recorded_across_the_course": sum(
            int(cell.get("risk_refusals") or 0) for cell in report["cells"]),
        "why_that_is_zero": ("strict refuses with relation_not_positive, "
                             "which is not a budget reason, so no probe "
                             "reached the risk-refusal ledger"),
        "not_re_run": ("the arms were not re-run; run2's numbers still carry "
                       "the strict rule and are reported as such"),
    }


def why_the_scope_proposals_failed(report: Mapping[str, Any],
                                  ) -> dict[str, Any]:
    """Each NO_FEASIBLE_THRESHOLD, attributed to what actually stopped it.

    DEV-AUTO-1 attributed all eight to the frozen bin grid -- "only 0.0 is a
    usable cut for the six fraction-type features".  The grid is not one
    tuple, the three features Slow actually named carry four distinct edges
    each, and a cut that moves the treated count and then fails a risk line is
    a risk answer, not a grid answer.  This separates the two.
    """
    from evaluation.main_protocol_p4 import scope_threshold_tool as tool
    grid = {name: list(edges) for name, edges in tool.frozen_bins().items()}
    rows: list[dict[str, Any]] = []
    for step in report["revision_steps"]:
        for evaluation in step.get("evaluations") or ():
            if evaluation.get("kind") != "scope_clause":
                continue
            calibration = evaluation.get("calibration") or {}
            tried = list(calibration.get("candidates_tried") or ())
            feature = calibration.get("feature")
            treated = {row.get("treated") for row in tried}
            distinct_cuts = len(treated)
            failed: dict[str, int] = {}
            for row in tried:
                for name, ok in (row.get("lines") or {}).items():
                    if not ok:
                        failed[name] = failed.get(name, 0) + 1
            if not tried:
                cause = "NO_EDGE_WAS_TRIED"
            elif distinct_cuts <= 1:
                cause = "THE_GRID: every edge selected the same series"
            else:
                cause = ("RISK: the edges moved the treated set and every one "
                         "of them failed a risk line")
            rows.append({
                "position": step["position"],
                "feature": feature,
                "direction": calibration.get("direction"),
                "edges_available_in_the_grid": grid.get(str(feature)),
                "edges_tried": [row.get("threshold") for row in tried],
                "treated_at_each_edge": [row.get("treated") for row in tried],
                "distinct_treated_counts": distinct_cuts,
                "lines_that_failed_and_how_often": failed,
                "attributed_to": cause,
            })
    causes: dict[str, int] = {}
    for row in rows:
        key = row["attributed_to"].split(":")[0]
        causes[key] = causes.get(key, 0) + 1
    return {
        "what_DEV_AUTO_1_said": ("all eight died at NO_FEASIBLE_THRESHOLD "
                                 "'for that instrument reason' -- the frozen "
                                 "bin grid"),
        "the_grid_is_not_one_tuple": grid,
        "attribution": causes,
        "per_proposal": rows,
    }


# ---------------------------------------------------------------------------
# 5. responsiveness -- an observation, not a causal claim
# ---------------------------------------------------------------------------

def responsiveness(report: Mapping[str, Any]) -> dict[str, Any]:
    def shape(step: Mapping[str, Any]) -> list[dict[str, Any]]:
        out = []
        for proposal in step.get("proposals") or ():
            if proposal.get("kind") == "workflow_program":
                out.append({"kind": "workflow",
                            "ops": [str(s.get("op"))
                                    for s in proposal["workflow_program"]]})
            elif proposal.get("kind") == "scope_clause":
                out.append({"kind": "scope",
                            "ops": ["%s %s" % (proposal["scope_clause"].get(
                                "feature"), proposal["scope_clause"].get("op"))]})
        return out

    rows, previous, previous_outcomes = [], None, None
    counts = {"operator_substituted": 0, "step_added_or_removed": 0,
              "scope_clause_changed": 0, "kind_switched": 0,
              "repeated_exactly": 0, "abstained": 0, "first_step": 0}
    for step in report["revision_steps"]:
        now = shape(step)
        outcomes = [e.get("outcome") for e in step.get("evaluations") or ()]
        row = {"position": step["position"], "outcome": step["outcome"],
               "proposals": now, "evaluation_outcomes": outcomes,
               "previous_failure": previous_outcomes}
        changes: list[str] = []
        if step["outcome"] == "ABSTAINED":
            changes = ["abstained"]
        elif previous is None:
            changes = ["first_step"]
        elif ({json.dumps(p, sort_keys=True) for p in previous}
              == {json.dumps(p, sort_keys=True) for p in now}):
            changes = ["repeated_exactly"]
        else:
            # More than one thing can change between two calls, so the label is
            # a set.  Collapsing it to one name is what made "10 scope
            # changes" out of steps that also swapped an operator.
            def side(rows_, kind):
                return {tuple(r["ops"]) for r in rows_ if r["kind"] == kind}
            before_w, after_w = side(previous, "workflow"), side(now, "workflow")
            before_s, after_s = side(previous, "scope"), side(now, "scope")
            if before_s != after_s:
                changes.append("scope_clause_changed")
            if before_w != after_w:
                lengths_before = {len(o) for o in before_w}
                lengths_after = {len(o) for o in after_w}
                changes.append("step_added_or_removed"
                               if lengths_before != lengths_after
                               else "operator_substituted")
            if bool(before_w) != bool(after_w) or bool(before_s) != bool(after_s):
                changes.append("kind_switched")
        row["changes"] = changes
        for name in changes:
            counts[name] += 1
        rows.append(row)
        if now:
            previous, previous_outcomes = now, outcomes
    counts["consecutive_proposing_pairs"] = sum(
        1 for r in rows if r["changes"] and r["changes"] != ["first_step"]
        and r["changes"] != ["abstained"])
    return {
        "what_this_is": ("a responsiveness observation: what the next call "
                         "named after the previous one failed.  It is not "
                         "evidence of causal learning, and no LLM was asked to "
                         "judge it"),
        "counts": counts, "rows": rows,
    }


# ---------------------------------------------------------------------------
# the package
# ---------------------------------------------------------------------------

def build() -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    clock = time.time()
    report = json.loads((ART / "dev_auto1_skill_revision__run2.json").read_text(
        encoding="utf-8"))
    audit = json.loads(
        (ART / "dev_auto1_blocked_candidates__run2.json").read_text(
            encoding="utf-8"))
    course = _course()
    store = m_r0k._load_store()
    # The arm is part of the cache key: a reading is never shared across
    # arms.  This package reads the one development arm the course ran.
    cache = runner.ReplayPredictionCache(live.ARM)
    # The ceiling is on fits actually paid.  A reservation guards the spend
    # before it happens; a reservation whose build then faulted without
    # touching a Consumer bought nothing and is not carried forward, or every
    # re-read of an unevaluable face would spend the package's budget twice.
    already = 0
    if FIT_LEDGER.is_file():
        already = int(json.loads(FIT_LEDGER.read_text(encoding="utf-8")).get(
            "physical_fits_spent_by_this_package") or 0)
    entries_before = len(store)
    spent = {"reserved": 0, "carried_in": already}

    def spend(n: int) -> None:
        total = already + max(spent["reserved"], cache.physical_fits) + int(n)
        if total > MAX_NEW_FITS:
            raise Ceiling("%d new fits would pass this package's ceiling of %d "
                          "(%d were already paid by an earlier run of it)"
                          % (total, MAX_NEW_FITS, already))
        if time.time() - clock > MAX_WALL_SECONDS:
            raise Ceiling("the two-hour wall clock is spent")
        spent["reserved"] += int(n)

    ancestor_support = _ancestor_cells(course, store, SUPPORT_FACE)
    ancestor_delayed = _ancestor_cells(course, store, DELAYED_FACE)
    reconstruction = _check_the_parent_reconstruction(report, course,
                                                      ancestor_support)
    if reconstruction["mismatches"]:
        return {"stage": "DEV_AUTO1R_REPAIR", "status": "BLOCKED",
                "why": "the parent reconstruction does not reproduce the "
                       "recorded parent summaries",
                "parent_reconstruction": reconstruction}

    executed = _executed_policy_cells(report, course, R.REVISING_ARM,
                                      SUPPORT_FACE)
    proposals = recompute_the_proposals(report, course, ancestor_support,
                                        executed)
    counts = restate_the_counts(proposals)
    decisions = frame_decisions(proposals)
    steps_by_label = _blocked_program_steps()

    pos23 = close_the_pos23_gap(course, store, cache, spend)

    frames: dict[str, Any] = {}
    validated: dict[str, Any] = {}
    for frame, name in (("A_absolute_historical_screen", "A"),
                        ("B_relative_reference", "B"),
                        ("C_forward_validation_first", "C")):
        queue = _queue(decisions, frame)
        results = forward_validate(queue, course, store, cache, spend,
                                   steps_by_label)
        frames[name] = {
            "rule": frame,
            "proposals_admitted": sum(1 for row in decisions
                                      if row[frame]["enters_verification"]),
            "distinct_programs_queued": len(queue),
            "queue": queue,
        }
        validated[name] = results
        frames[name]["validation"] = _validation_counts(results)

    return {
        "stage": "DEV_AUTO1R_REPAIR",
        "package": "DEV-AUTO-1R: one comparator, three acceptance frames",
        "what_this_is_not": [
            "not a new experiment and not a re-run of the two arms",
            "not a claim about the +144 face, sealed material or any "
            "TARGET_HELD_IN unit -- none was read",
            "not a change to any Consumer, DSL, risk threshold, risk "
            "denominator or Scope grid",
        ],
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "sources": {
            "run": "artifacts/main_protocol/dev_auto1_skill_revision__run2.json",
            "audit": ("artifacts/main_protocol/"
                      "dev_auto1_blocked_candidates__run2.json"),
            "store": str(m_r0k.STORE.name),
        },
        "parent_reconstruction": reconstruction,
        "comparator": {
            "function": ("dev_auto1_revision.compare_on_a_common_denominator"),
            "states": [rev.READ, rev.RAW_FALLBACK, rev.UNKNOWN],
            "raw_fallback_authority": rev._RAW_FALLBACK_AUTHORITY,
        },
        "recomputed_counts": counts,
        "position_23_support_gap": pos23,
        "the_audit_rechecked": recheck_the_audit(audit, course, pos23),
        "reference_declaration": REFERENCE_DECLARATION,
        "frames": frames,
        "forward_validation": validated,
        "what_stopped_fast": what_stopped_fast(report),
        "why_the_scope_proposals_failed": why_the_scope_proposals_failed(report),
        "responsiveness": responsiveness(report),
        "proposals": proposals,
        "frame_decisions": decisions,
        "actual_cost": {
            "llm_calls": 0,
            "new_physical_consumer_fits": already + int(cache.physical_fits),
            "physically_measured_this_run": int(cache.physical_fits),
            "paid_by_an_earlier_run_of_this_package": already,
            "reserved_this_run_before_the_spend": spent["reserved"],
            "store_entries_before": entries_before,
            "store_entries_after": len(store),
            "why_two_numbers": (
                "the ceiling is enforced on a reservation of %d fits per "
                "entry, checked before the spend; what is billed is what the "
                "cache actually paid.  An entry reused from the store costs "
                "nothing, and a reservation whose build then faulted before "
                "touching a Consumer bought nothing" % FITS_PER_ENTRY),
            "ceiling": MAX_NEW_FITS,
            "wall_seconds": round(time.time() - clock, 1),
            "wall_ceiling": MAX_WALL_SECONDS,
            "billed_to": "DEV-AUTO-1R",
        },
        "boundary": {
            "evaluation_face_reads": 0,
            "sealed_reads": 0,
            "target_held_in_reads": 0,
            "risk_constants_touched": 0,
            "scope_grid_touched": 0,
            "historical_receipts_overwritten": 0,
        },
    }


def _validation_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {"entered": len(results), "support_admitted_at_least_once": 0,
           "validation_passed": 0, "no_window_passed": 0,
           "attempts_exhausted": 0, "not_evaluated": 0, "no_later_unit": 0}
    for row in results:
        if any(a.get("support_admits") for a in row["attempts"]):
            out["support_admitted_at_least_once"] += 1
        outcome = row["outcome"]
        out["validation_passed" if outcome == "VALIDATION_PASSED" else
            "no_window_passed" if outcome == "NO_WINDOW_PASSED" else
            "attempts_exhausted" if outcome
            == "VERIFICATION_ATTEMPTS_EXHAUSTED" else
            "no_later_unit" if outcome == "NO_LATER_UNIT" else
            "not_evaluated"] += 1
    out["promotion"] = ("not counted here: promotion additionally requires "
                        "Fast to deploy the program in the passing unit")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    result = build()
    cost = result.get("actual_cost") or {}
    if cost:
        FIT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        FIT_LEDGER.write_text(json.dumps({
            "package": "DEV-AUTO-1R",
            "physical_fits_spent_by_this_package":
                int(cost.get("new_physical_consumer_fits") or 0),
            "ceiling": MAX_NEW_FITS}, ensure_ascii=False), encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(drafts._plain(result), ensure_ascii=False,
                              indent=1, default=str), encoding="utf-8")
    cost = result.get("actual_cost") or {}
    print("%s | new fits %s/%s | llm %s"
          % (result.get("status") or "OK", cost.get("new_physical_consumer_fits"),
             cost.get("ceiling"), cost.get("llm_calls")))
    for name, block in (result.get("frames") or {}).items():
        print("  frame %s: admitted %s proposals, %s distinct programs, %s"
              % (name, block["proposals_admitted"],
                 block["distinct_programs_queued"],
                 json.dumps(block["validation"], ensure_ascii=False)[:200]))
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
