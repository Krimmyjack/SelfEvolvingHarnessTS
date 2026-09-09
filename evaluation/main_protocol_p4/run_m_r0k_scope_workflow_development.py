"""M-R0k: what a Scope edit and a Workflow edit each have to win, on development.

The first of the three packages in the execution plan's 4.5.  It is a
**development comparison**, not an evolution experiment and not the old
appendix-D M-W: every unit here is already exposed, nothing is activated, no
lifecycle counter moves and no deployment right is granted or withdrawn.

Three questions, and only three
-------------------------------
1. Among the legal candidates this package names, is there a modification that
   keeps the gain while satisfying the risk and coverage lines?
2. Can a selector that sees **only pre-boundary evidence** find it, or should it
   answer ``NO_REVISION``?
3. What is the remaining opportunity on each line -- more utility, restored
   eligibility, or only less treatment?

Fixed geometry
--------------
* forward / A5-online, ancestor ``outlier_mad({})`` under
  ``local_robust_z_peak >= 3.0``; the selection boundary is k1 / p04.
* The 21 development units M-R0j exposed.  The five ``TARGET_HELD_IN`` units are
  excluded by exposure name and never read; so is the +144 evaluation face.
* Both faces: the unit's own origin (Support, evidence) and origin+48 (delayed,
  where the authoritative gate lives).
* Everything is scored over the **full served population**; an out-of-scope
  series carries the raw prediction and scores exactly zero.  A narrowing is
  therefore scored on its ancestor's denominator, which is the alignment 4.5(a)
  asks for.
* The first five units (positions 0-4) are the selector's evidence; the sixteen
  after the boundary are reported separately, so a degradation caused by the
  five pre-boundary units is never charged to the later ones.

The parent version
------------------
Whether the parent was available is read off the **rights snapshot at the
boundary**: the K0 card ``fast_winner_forecast_ridge_smase_outlier_mad`` is in
the active snapshot at every position and was never revoked.  The Draft under
revision carries ``deployable=false``; that is the Draft's state and is not read
as the K0 ancestor having lost rights.  No gate result from after the boundary is
used to switch the comparison target.

Time discipline
---------------
The three Workflow candidates were named by the researcher **after** the course
was read.  They are therefore a **development selection replay**: legitimate for
measuring the space and what a selector would do in it, never a claim that k1
proposed them.  This is stated in the receipt next to every Workflow number.

Cost
----
0 LLM calls.  The physical Consumer-fit ceiling is 400 and every fit counts,
including smoke, retries and fits spent inside a call that then raises.  Each
new ``(unit, face, program)`` cache entry costs two fits and is re-masked for
every Scope, so the whole Scope line costs what one program's readings cost.

Run:  python -m evaluation.main_protocol_p4.run_m_r0k_scope_workflow_development
"""
from __future__ import annotations

import itertools
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import audit_m_r0b_revision_opportunity as opp
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import scope_narrowing_preflight as preflight
from evaluation.main_protocol_p4 import scope_threshold_tool as tool
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as smoke

ART = base.ART
MAX_FITS = smoke.MAX_FITS
MATERIAL = float(contract.RISK["material"])
MAX_HARMED = float(contract.RISK["max_harmed_fraction"])
MAX_HARM = float(contract.RISK["max_single_series_harm"])
FLOOR = int(contract.RISK["min_treated"])
DELAYED = int(runner.DELAYED_OFFSET)
BOUNDARY = int(live.BOUNDARY_POSITION)

ANCESTOR_SCOPE = smoke.ANCESTOR_SCOPE
ANCESTOR_PROGRAM = smoke.ANCESTOR_PROGRAM
WORKFLOW_CANDIDATES = smoke.WORKFLOW_CANDIDATES
FACES = ("support_face", "delayed_face")

#: The random-retention reference draws this many subsets per (unit, face,
#: coverage).  The expectation is analytic; the draws exist only to report the
#: gate's pass **rate**, which is not the gate applied to the expectation.
RANDOM_DRAWS = 4000
RANDOM_SEED = 20260906

STORE_DIR = Path(os.environ.get("M_R0K_STORE_DIR")
                 or (Path(__file__).resolve().parents[2] / "_scratch"))
STORE = STORE_DIR / "m_r0k_prediction_store.json"


class InstrumentFault(RuntimeError):
    """The shared scorer disagrees with itself.  Stop the affected fits."""


# ---------------------------------------------------------------------------
# the prediction store: two fits per (unit, face, program), re-masked for free
# ---------------------------------------------------------------------------

def _entry_key(program: str, position: int, face: str) -> str:
    return "%s|%d|%s" % (program, position, face)


def _load_store() -> dict[str, Any]:
    if STORE.is_file():
        try:
            return json.loads(STORE.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def _save_store(store: Mapping[str, Any]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(store, ensure_ascii=False, default=str),
                     encoding="utf-8")


def _reading_from_entry(entry: Mapping[str, Any],
                        resolved: Sequence[str] | frozenset[str],
                        ) -> dict[str, Any]:
    """``ReplayPredictionCache.reading``'s arithmetic, off a stored entry.

    Reproduced rather than called so an entry can be scored after the process
    that fitted it is gone.  Phase 1 asserts the two agree bit for bit on every
    entry it builds; a disagreement is an instrument fault and stops the fits.
    """
    if not entry.get("verifier_passed"):
        return {"status": "WINDOW_VERIFIER_REJECTED",
                "why": "the window verifier refused this program at this face"}
    selected = set(str(uid) for uid in (resolved or ()))
    reached = sorted(selected & set(entry.get("degenerate_uids") or ()))
    if reached:
        return {"status": "SERVING_CONTEXT_DEGENERATE",
                "why": ("preparing the served context flattened %d scoped "
                        "series (%s)" % (len(reached), ", ".join(reached[:6]))),
                "series": reached}
    uids = [str(uid) for uid in entry["eval_uids"]]
    raw = np.asarray(entry["raw_per_view"], dtype=np.float64)
    program = np.asarray(entry["program_per_view"], dtype=np.float64)
    mask = np.array([uid in selected for uid in uids], dtype=bool)
    per_view = np.where(mask, program, raw)
    gains = raw - per_view
    return {
        "status": "READ",
        "identity": not selected,
        "treated": int(mask.sum()),
        "served": len(uids),
        "per_series_gain": {uid: round(float(value), 6)
                            for uid, value in zip(uids, gains)},
        "exact_gain": {uid: float(value) for uid, value in zip(uids, gains)},
        "aggregate_gain": round(float(gains.mean()), 6),
        "harmed_fraction": round(float((gains < -MATERIAL).mean()), 4),
        "harmed_series": int((gains < -MATERIAL).sum()),
        "max_single_series_harm": round(max(0.0, float(-gains.min())), 6),
        "mean_smase": float(per_view.mean()),
        "static_mean_smase": float(raw.mean()),
    }


def _gate(reading: Mapping[str, Any]) -> dict[str, Any]:
    if reading.get("status") != "READ":
        return {"passes": False, "failed_lines": ["not_readable"],
                "why": reading.get("status")}
    return runner.authoritative_gate(reading)


def _served_violations(reading: Mapping[str, Any]) -> list[str]:
    """The screen's risk lines, on the served denominator this package uses."""
    if reading.get("status") != "READ":
        return ["not_readable"]
    failed = []
    if float(reading["aggregate_gain"]) <= MATERIAL:
        failed.append("aggregate_not_material")
    if float(reading["harmed_fraction"]) > MAX_HARMED:
        failed.append("harmed_fraction")
    if float(reading["max_single_series_harm"]) > MAX_HARM:
        failed.append("single_series_harm")
    return failed


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------

def _population(doc: Mapping[str, Any]) -> tuple[list[dict[str, Any]],
                                                 list[dict[str, Any]]]:
    kept, excluded = [], []
    for cell in base.cells(doc, live.ARM):
        unit = dict(cell["unit"])
        exposure = list(unit.get("exposure") or ())
        row = {"position": int(cell["position"]), "unit": unit,
               "exposure": exposure,
               "side": ("before_the_boundary"
                        if int(cell["position"]) <= BOUNDARY
                        else "after_the_boundary")}
        if any(name in smoke.EXCLUDED_EXPOSURES for name in exposure):
            excluded.append({**row, "why": "held for the Target, not development"})
        elif any(name in smoke.DEVELOPMENT_EXPOSURES for name in exposure):
            kept.append(row)
        else:
            excluded.append({**row, "why": "exposure is not a development one"})
    return kept, excluded


def _faces_for(ctx: Any) -> dict[str, int]:
    return {"support_face": int(ctx.origin),
            "delayed_face": int(ctx.face_origin(DELAYED))}


# ---------------------------------------------------------------------------
# phase 1 / 3: fit one program on every unit and face
# ---------------------------------------------------------------------------

def _build_program(label: str, steps: Sequence[tuple], population,
                   cache: runner.ReplayPredictionCache,
                   store: dict[str, Any]) -> dict[str, Any]:
    """Two fits per (unit, face) unless the store already holds the entry."""
    built, reused, refused, faults = 0, 0, [], []
    for row in population:
        ctx = opp._unit_ctx(row["unit"])
        for face, origin in _faces_for(ctx).items():
            key = _entry_key(label, row["position"], face)
            if key in store:
                reused += 1
                continue
            before = cache.physical_fits
            cache_key = cache.key(ctx.unit, origin,
                                  runner.forecast_p4._config(origin), steps)
            if (cache_key not in cache._entries
                    and cache.physical_fits + 2 > MAX_FITS):
                refused.append({"position": row["position"], "face": face,
                                "why": "REFUSED_AT_THE_CEILING",
                                "spent": cache.physical_fits})
                continue
            try:
                entry = cache._build(ctx, origin, tuple(steps))
                cache._entries[cache_key] = entry
            except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                faults.append({"position": row["position"], "face": face,
                               "fault": "%s: %s" % (type(exc).__name__,
                                                    str(exc)[:200]),
                               "fits_spent_before_the_fault":
                                   cache.physical_fits - before})
                continue
            spent = cache.physical_fits - before
            store[key] = {
                "program": label, "position": row["position"], "face": face,
                "origin": origin,
                "verifier_passed": bool(entry.get("verifier_passed")),
                "eval_uids": list(entry.get("eval_uids") or ()),
                "raw_per_view": [float(v)
                                 for v in (entry.get("raw_per_view") or ())],
                "program_per_view": [float(v) for v in
                                     (entry.get("program_per_view") or ())],
                "degenerate_uids": list(entry.get("degenerate_uids") or ()),
                "physical_fits": int(spent),
            }
            built += 1
            # The shared scorer, checked against itself while both are in hand.
            if entry.get("verifier_passed"):
                mine = _reading_from_entry(store[key], frozenset(ctx.eval_uids))
                theirs = cache.reading(ctx, origin, tuple(steps),
                                       frozenset(ctx.eval_uids))
                for field in ("treated", "served", "aggregate_gain",
                              "harmed_fraction", "max_single_series_harm"):
                    if repr(mine[field]) != repr(theirs[field]):
                        raise InstrumentFault(
                            "the stored re-mask disagrees with the production "
                            "cache on %s at position %d %s: %r vs %r"
                            % (field, row["position"], face, mine[field],
                               theirs[field]))
            _save_store(store)
    return {"program": label,
            "steps": [{"op": op, "params": dict(params)} for op, params in steps],
            "entries_built": built, "entries_reused_from_the_store": reused,
            "refused_at_the_ceiling": refused, "faults": faults,
            "physical_fits_after": cache.physical_fits}


def _bank_agrees_with_the_store(bank_rows: Sequence[Mapping[str, Any]],
                                population, store: Mapping[str, Any]
                                ) -> dict[str, Any]:
    """The k1 bank and this package's readings must be the same measurement.

    The bank rows the threshold tool calibrates on are the deployed policy's
    per-series gains: exactly zero outside the ancestor Scope, the program's
    gain inside it.  If those numbers disagreed with the re-masked store, the
    selector would be ranking candidates against evidence the evaluation does
    not reproduce, and every column in this package would be comparing two
    different experiments.  Checked series by series, exactly.
    """
    by_unit: dict[tuple[str, int], dict[str, float]] = {}
    for row in bank_rows:
        unit = dict(row.get("unit") or {})
        key = (str(unit.get("block")), int(unit.get("origin")))
        by_unit.setdefault(key, {})[str(row["series"])] = float(row["gain"])
    compared, mismatched, units = 0, [], 0
    scope_set_matches = True
    for unit_row in population:
        key = (str(unit_row["unit"]["block"]), int(unit_row["unit"]["origin"]))
        entry = store.get(_entry_key("ANCESTOR", unit_row["position"],
                                     "support_face"))
        if key not in by_unit or entry is None or not entry.get(
                "verifier_passed"):
            continue
        ctx = opp._unit_ctx(unit_row["unit"])
        resolved = set(ctx.resolve(ANCESTOR_SCOPE, ctx.origin))
        raw = np.asarray(entry["raw_per_view"], dtype=np.float64)
        program = np.asarray(entry["program_per_view"], dtype=np.float64)
        gains = {str(uid): float(value) for uid, value
                 in zip(entry["eval_uids"], raw - program)}
        units += 1
        non_zero = {uid for uid, value in by_unit[key].items() if value != 0.0}
        if non_zero != resolved:
            scope_set_matches = False
        for uid in sorted(resolved):
            if uid not in by_unit[key] or uid not in gains:
                mismatched.append({"unit": key, "series": uid,
                                   "why": "series missing on one side"})
                continue
            compared += 1
            if abs(by_unit[key][uid] - gains[uid]) > 1e-6:
                mismatched.append({"unit": key, "series": uid,
                                   "bank": by_unit[key][uid],
                                   "store": gains[uid]})
    return {
        "check": "the k1 bank and the re-masked store are the same measurement",
        "units_compared": units,
        "series_compared_inside_the_ancestor_scope": compared,
        "mismatches": mismatched,
        "the_banks_non_zero_set_equals_the_resolved_scope": scope_set_matches,
        "passed": not mismatched and scope_set_matches,
        "why_it_matters": ("the selector ranks on bank evidence and the "
                           "comparison scores on the store; if they differed "
                           "the columns would be two different experiments"),
    }


def _raw_consistency(store: Mapping[str, Any], programs: Sequence[str]
                     ) -> dict[str, Any]:
    """The raw pipeline may not depend on the program.  Checked, not assumed."""
    mismatches = []
    by_face: dict[tuple[int, str], list[str]] = {}
    for key, entry in store.items():
        by_face.setdefault((entry["position"], entry["face"]), []).append(key)
    for (position, face), keys in sorted(by_face.items()):
        vectors = {store[k]["program"]: store[k]["raw_per_view"] for k in keys
                   if store[k].get("verifier_passed")}
        if len(vectors) < 2:
            continue
        names = sorted(vectors)
        first = vectors[names[0]]
        for name in names[1:]:
            if [repr(v) for v in vectors[name]] != [repr(v) for v in first]:
                mismatches.append({"position": position, "face": face,
                                   "programs": [names[0], name]})
    return {"check": "the raw pipeline is identical across programs",
            "faces_compared": len(by_face), "mismatches": mismatches,
            "passed": not mismatches,
            "why_it_matters": ("raw is the shared scorer's reference and the "
                               "out-of-scope fallback; if it moved with the "
                               "program the pairing would be wrong")}


# ---------------------------------------------------------------------------
# phase 2: the legal Scope candidate space, enumerated with its calibration
# ---------------------------------------------------------------------------

def _scope_of(clause: Mapping[str, Any] | None) -> dict[str, Any]:
    predicate = [dict(c) for c in ANCESTOR_SCOPE["predicate"]]
    if clause is not None:
        predicate = predicate + [dict(clause)]
    return {"scope_type": "serving_series_predicate", "predicate": predicate}


def _enumerate_scope_candidates(rows: Sequence[Mapping[str, Any]],
                                processed) -> dict[str, Any]:
    """Every (feature, direction) in the frozen vocabulary, with its product.

    Nothing is added: the vocabulary, the frozen bin edges, ``calibrate``'s
    widest-feasible-edge rule and ``validate_narrowing`` are the production
    ones.  A pair that ``calibrate`` refuses is recorded with the edges it
    tried, so the space is reported whole rather than only where it succeeded.
    """
    existing = [dict(c) for c in ANCESTOR_SCOPE["predicate"]]
    table: list[dict[str, Any]] = []
    for feature in contract.SCOPE_CLASS["vocabulary"]:
        for direction in tool.DIRECTIONS:
            row: dict[str, Any] = {"feature": str(feature),
                                   "direction": str(direction)}
            try:
                chosen = tool.calibrate(feature=str(feature),
                                        direction=str(direction), rows=rows,
                                        existing_clauses=existing)
            except tool.NoFeasibleThreshold as exc:
                row.update({"outcome": "NO_FEASIBLE_THRESHOLD",
                            "candidates_tried": exc.to_dict()["candidates_tried"]})
                table.append(row)
                continue
            except Exception as exc:  # noqa: BLE001 - recorded, never relaxed
                row.update({"outcome": "REFUSED",
                            "why": "%s: %s" % (type(exc).__name__,
                                               str(exc)[:200])})
                table.append(row)
                continue
            clause = dict(chosen["clause"])
            scope = _scope_of(clause)
            verdict = preflight.validate_narrowing(ANCESTOR_SCOPE, scope,
                                                   root=None)
            coverage = []
            for ctx in processed:
                try:
                    treated = len(ctx.resolve(scope, ctx.origin))
                except Exception as exc:  # noqa: BLE001
                    coverage.append({"origin": ctx.origin,
                                     "unresolvable": str(exc)[:120]})
                    continue
                coverage.append({"origin": ctx.origin, "treated": treated,
                                 "at_or_above_floor": treated >= FLOOR})
            row.update({
                "outcome": "CALIBRATED",
                "clause": clause,
                "scope": scope,
                "threshold_is_a_frozen_bin_edge": True,
                "bin_edges": chosen["bin_edges"],
                "tie_break": chosen["tie_break"],
                "pooled_treated_rows": chosen["treated"],
                "bank_rows_in_scope_before": chosen["bank_rows_in_scope_before"],
                "pooled_reading_on_treated_rows": chosen["reading"],
                "preflight": {"accepted": bool(getattr(verdict, "accepted",
                                                       False)),
                              "detail": (verdict.to_dict()
                                         if hasattr(verdict, "to_dict")
                                         else str(verdict))},
                "historical_coverage_before_the_boundary": coverage,
                "cells_at_or_above_floor": sum(
                    1 for c in coverage if c.get("at_or_above_floor")),
            })
            table.append(row)
    calibrated = [r for r in table if r.get("outcome") == "CALIBRATED"]
    return {
        "pairs_enumerated": len(table),
        "vocabulary": list(contract.SCOPE_CLASS["vocabulary"]),
        "directions": list(tool.DIRECTIONS),
        "calibrated": len(calibrated),
        "refused": len(table) - len(calibrated),
        "table": table,
        "space": ("the ancestor plus at most one calibrated clause: the "
                  "production narrowing path adds one clause per revision and "
                  "the Scope class allows at most two, so this enumeration is "
                  "the whole legal one-step space from this ancestor"),
        "nothing_added": {"features": 0, "thresholds": 0, "operators": 0,
                          "bins": "frozen", "rules": "production"},
    }


# ---------------------------------------------------------------------------
# scoring a policy over the population
# ---------------------------------------------------------------------------

def _policy_rows(label: str, program_label: str,
                 scope: Mapping[str, Any] | None, population,
                 store: Mapping[str, Any], ancestor_key: str,
                 ) -> list[dict[str, Any]]:
    """One policy on every unit and face, paired against the ancestor."""
    rows = []
    for unit_row in population:
        ctx = opp._unit_ctx(unit_row["unit"])
        entry_faces = {}
        for face, origin in _faces_for(ctx).items():
            key = _entry_key(program_label, unit_row["position"], face)
            anc_key = _entry_key(ancestor_key, unit_row["position"], face)
            entry = store.get(key)
            anc_entry = store.get(anc_key)
            if entry is None or anc_entry is None:
                entry_faces[face] = {"status": "NO_STORED_READING"}
                continue
            try:
                resolved = frozenset(ctx.resolve(scope, origin)) if scope \
                    else frozenset(ctx.eval_uids)
            except Exception as exc:  # noqa: BLE001
                entry_faces[face] = {"status": "SCOPE_UNRESOLVABLE",
                                     "why": str(exc)[:160]}
                continue
            anc_resolved = frozenset(ctx.resolve(ANCESTOR_SCOPE, origin))
            reading = _reading_from_entry(entry, resolved)
            ancestor = _reading_from_entry(anc_entry, anc_resolved)
            face_row: dict[str, Any] = {
                "origin": origin,
                "resolved": sorted(resolved),
                "ancestor_resolved": sorted(anc_resolved),
                "reading": {k: v for k, v in reading.items()
                            if k not in ("exact_gain",)},
                "ancestor_reading": {k: v for k, v in ancestor.items()
                                     if k not in ("exact_gain",
                                                  "per_series_gain")},
                "gate": _gate(reading),
                "ancestor_gate": _gate(ancestor),
                "served_denominator_violations": _served_violations(reading),
            }
            if reading.get("status") == "SERVING_CONTEXT_DEGENERATE":
                # What the pair would have delivered if the degenerate series
                # were dropped.  Recorded because it sizes the opportunity the
                # legality rule costs; it is NOT a policy this protocol may
                # deploy, because dropping them silently would make the Scope
                # mean something other than what it declared.
                trimmed = _reading_from_entry(
                    entry, resolved - set(entry.get("degenerate_uids") or ()))
                face_row["diagnostic_if_the_degenerate_series_were_dropped"] = {
                    "not_a_production_legal_policy": True,
                    "reading": {k: v for k, v in trimmed.items()
                                if k not in ("exact_gain", "per_series_gain")},
                }
            if reading.get("status") == "READ" and ancestor.get("status") == "READ":
                mine = reading["exact_gain"]
                theirs = ancestor["exact_gain"]
                differing = [uid for uid in sorted(set(mine) | set(theirs))
                             if repr(float(mine.get(uid, 0.0)))
                             != repr(float(theirs.get(uid, 0.0)))]
                excluded = sorted(set(anc_resolved) - set(resolved))
                added = sorted(set(resolved) - set(anc_resolved))
                harmful = [u for u in excluded if theirs.get(u, 0.0) < -MATERIAL]
                helpful = [u for u in excluded if theirs.get(u, 0.0) > MATERIAL]
                neutral = [u for u in excluded
                           if abs(theirs.get(u, 0.0)) <= MATERIAL]
                face_row.update({
                    "behaviour_identical": not differing,
                    "series_scored_differently": len(differing),
                    "delta_aggregate_gain": round(
                        reading["aggregate_gain"] - ancestor["aggregate_gain"],
                        6),
                    "delta_harmed_series": (reading["harmed_series"]
                                            - ancestor["harmed_series"]),
                    "delta_max_single_series_harm": round(
                        reading["max_single_series_harm"]
                        - ancestor["max_single_series_harm"], 6),
                    "delta_treated": reading["treated"] - ancestor["treated"],
                    "exclusion_account": {
                        "excluded_from_the_ancestor": len(excluded),
                        "added_beyond_the_ancestor": len(added),
                        "excluded_and_harmful": len(harmful),
                        "excluded_and_positive": len(helpful),
                        "excluded_and_neutral": len(neutral),
                        "exclusion_precision": (
                            round(len(harmful) / len(excluded), 4)
                            if excluded else None),
                        "cost_of_excluding_the_positive_ones": round(
                            sum(float(theirs[u]) for u in helpful), 6),
                        "harm_avoided_by_the_exclusions": round(
                            -sum(float(theirs[u]) for u in harmful), 6),
                    },
                })
            else:
                face_row.update({"behaviour_identical": None,
                                 "delta_aggregate_gain": None})
            entry_faces[face] = face_row
        rows.append({"policy": label, "position": unit_row["position"],
                     "unit": unit_row["unit"], "side": unit_row["side"],
                     "exposure": unit_row["exposure"], "faces": entry_faces})
    return rows


def _executable_gain(face_row: Mapping[str, Any]) -> float:
    """What the policy actually delivers on this face.

    An illegal ``(program, scope)`` pair here is not deployable, and the
    recorded legal fallback in that situation is the identity / raw pipeline --
    exactly what the course did on the units where nothing was deployed.  So an
    unreadable face contributes exactly 0.0 rather than being dropped, and the
    count of such faces is reported next to the number.
    """
    reading = face_row.get("reading") or {}
    if reading.get("status") != "READ":
        return 0.0
    return float(reading["aggregate_gain"])


def _summarise(rows: Sequence[Mapping[str, Any]], face: str,
               side: str | None = None) -> dict[str, Any]:
    chosen = [r for r in rows if side is None or r["side"] == side]
    faces = [r["faces"].get(face) or {} for r in chosen]
    readable = [f for f in faces if (f.get("reading") or {}).get("status")
                == "READ"]
    unreadable = [{"position": r["position"],
                   "origin": (r["faces"].get(face) or {}).get("origin"),
                   "status": ((r["faces"].get(face) or {}).get("reading")
                              or {}).get("status")
                   or (r["faces"].get(face) or {}).get("status"),
                   "why": ((r["faces"].get(face) or {}).get("reading")
                           or {}).get("why")}
                  for r in chosen
                  if ((r["faces"].get(face) or {}).get("reading") or {}).get(
                      "status") != "READ"]
    gains = [_executable_gain(f) for f in faces]
    deltas = [f["delta_aggregate_gain"] for f in faces
              if f.get("delta_aggregate_gain") is not None]
    differing = [f for f in faces if f.get("behaviour_identical") is False]
    gates = [bool((f.get("gate") or {}).get("passes")) for f in faces]
    failed_lines: dict[str, int] = {}
    for f in faces:
        for line in (f.get("gate") or {}).get("failed_lines") or ():
            failed_lines[line] = failed_lines.get(line, 0) + 1
    excluded_cost = sum(float((f.get("exclusion_account") or {}).get(
        "cost_of_excluding_the_positive_ones") or 0.0) for f in faces)
    excluded_harm = sum(float((f.get("exclusion_account") or {}).get(
        "harm_avoided_by_the_exclusions") or 0.0) for f in faces)
    excluded_total = sum(int((f.get("exclusion_account") or {}).get(
        "excluded_from_the_ancestor") or 0) for f in faces)
    excluded_harmful = sum(int((f.get("exclusion_account") or {}).get(
        "excluded_and_harmful") or 0) for f in faces)
    excluded_positive = sum(int((f.get("exclusion_account") or {}).get(
        "excluded_and_positive") or 0) for f in faces)
    return {
        "units": len(chosen),
        "faces_readable": len(readable),
        "faces_not_readable": unreadable,
        "mean_executable_aggregate_gain": (round(statistics.fmean(gains), 6)
                                           if gains else None),
        "unreadable_faces_score_zero": (
            "an illegal (program, scope) pair falls back to raw, the recorded "
            "legal fallback; it stays in the denominator at exactly 0.0"),
        "total_treated": sum(int((f.get("reading") or {}).get("treated") or 0)
                             for f in readable),
        "total_harmed_series": sum(
            int((f.get("reading") or {}).get("harmed_series") or 0)
            for f in readable),
        "worst_single_series_harm": (
            round(max(float((f.get("reading") or {}).get(
                "max_single_series_harm") or 0.0) for f in readable), 6)
            if readable else None),
        "gate_passes": sum(1 for g in gates if g),
        "gate_failed_line_counts": failed_lines,
        "behaviour_identical_to_the_ancestor": sum(
            1 for f in faces if f.get("behaviour_identical") is True),
        "behaviour_different": len(differing),
        "delta_vs_ancestor": {
            "n_with_a_paired_reading": len(deltas),
            "mean_over_every_paired_unit": (round(statistics.fmean(deltas), 6)
                                            if deltas else None),
            "mean_where_behaviour_differs": (
                round(statistics.fmean(
                    [f["delta_aggregate_gain"] for f in differing]), 6)
                if differing else None),
            "negative": sum(1 for d in deltas if d < 0),
            "positive": sum(1 for d in deltas if d > 0),
            "zero": sum(1 for d in deltas if d == 0),
            "harmed_series_net": sum(int(f.get("delta_harmed_series") or 0)
                                     for f in faces),
        },
        "exclusion_account": {
            "series_excluded_from_the_ancestor": excluded_total,
            "of_those_harmful": excluded_harmful,
            "of_those_positive": excluded_positive,
            "exclusion_precision": (round(excluded_harmful / excluded_total, 4)
                                    if excluded_total else None),
            "cost_of_excluding_the_positive_ones": round(excluded_cost, 6),
            "harm_avoided_by_the_exclusions": round(excluded_harm, 6),
            "net": round(excluded_harm - excluded_cost, 6),
        },
    }


# ---------------------------------------------------------------------------
# the selectors
# ---------------------------------------------------------------------------

def _pre_boundary_score(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The evidence a boundary-time selector is allowed to see.

    The Support face of the five processed units -- the face the bank was
    written on and the face ``outer_loop.screen`` replays.  Nothing after the
    boundary and no delayed reading is consulted.
    """
    faces = [(r["faces"].get("support_face") or {}) for r in rows
             if r["side"] == "before_the_boundary"]
    gains = [_executable_gain(f) for f in faces]
    applicable = [f for f in faces
                  if (f.get("reading") or {}).get("status") == "READ"
                  and int((f.get("reading") or {}).get("treated") or 0) >= FLOOR]
    violations = {}
    for f in applicable:
        for name in _served_violations(f.get("reading") or {}):
            violations[name] = violations.get(name, 0) + 1
    return {
        "face": "support_face (the five pre-boundary units)",
        "mean_aggregate_gain_over_the_served_population": (
            round(statistics.fmean(gains), 6) if gains else None),
        "cells_applicable_at_or_above_the_floor": len(applicable),
        "served_denominator_violations": violations,
        "screen_style_pass": bool(applicable and not violations),
    }


def _why_not_unified(preflight_ok: bool, pre: Mapping[str, Any]) -> str:
    if not preflight_ok:
        return "validate_narrowing refuses it"
    if not pre["cells_applicable_at_or_above_the_floor"]:
        return ("no pre-boundary cell reaches the coverage floor, so there is "
                "no applicable reading to test the risk lines on")
    return ("a risk line fails on the served denominator: %s"
            % sorted(pre["served_denominator_violations"]))


def _select(space: Sequence[Mapping[str, Any]], *, rule: str,
            ancestor_score: float) -> dict[str, Any]:
    """Rank a scored candidate space and decide, with NO_REVISION available."""
    feasible = [c for c in space if c["eligible"]]
    ranked = sorted(feasible, key=lambda c: (-float(c["score"]),
                                             -int(c.get("coverage") or 0),
                                             str(c["label"])))
    best = ranked[0] if ranked else None
    compared = math.isfinite(ancestor_score)
    material_gap = (round(float(best["score"]) - ancestor_score, 6)
                    if best is not None and compared else None)
    picked = (best["label"] if best is not None
              and float(best["score"]) >= ancestor_score + MATERIAL
              else "NO_REVISION")
    return {
        "rule": rule,
        "ancestor_score": (round(ancestor_score, 6) if compared else
                           "not used by this rule: it ranks modifications "
                           "without comparing them to the parent"),
        "candidates_considered": len(space),
        "candidates_eligible": len(feasible),
        "ranking": [{"label": c["label"], "score": round(float(c["score"]), 6),
                     "coverage": c.get("coverage"),
                     "eligible": c["eligible"],
                     "why_not": c.get("why_not")} for c in
                    sorted(space, key=lambda c: -float(c["score"]))],
        "best_eligible": (best["label"] if best else None),
        "best_eligible_score": (round(float(best["score"]), 6) if best else None),
        "margin_over_the_parent": material_gap,
        "material": MATERIAL,
        "picked": picked,
        "no_revision_is_a_first_class_option": True,
        "comparison_target": ("the parent version, which the boundary rights "
                              "snapshot shows was deployable"),
    }


# ---------------------------------------------------------------------------
# the random-retention reference
# ---------------------------------------------------------------------------

def _random_reference(rows: Sequence[Mapping[str, Any]], face: str,
                      store: Mapping[str, Any], ancestor_key: str,
                      population) -> dict[str, Any]:
    """Same coverage, chosen at random from the ancestor's own selection.

    The expectation is analytic: gains are additive over series and an
    unselected series scores exactly zero, so a random size-k subset of the
    ancestor's set A has expected aggregate ``(k/|A|) x ancestor_aggregate``.
    The **pass rate** of the authoritative gate is a different object and is
    estimated by drawing subsets, never by putting the expectation through the
    gate.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    per_unit = []
    for row in rows:
        f = row["faces"].get(face) or {}
        reading = f.get("reading") or {}
        if reading.get("status") != "READ":
            per_unit.append({"position": row["position"],
                             "status": reading.get("status")
                             or f.get("status")})
            continue
        anc_key = _entry_key(ancestor_key, row["position"], face)
        entry = store.get(anc_key)
        if entry is None:
            per_unit.append({"position": row["position"],
                             "status": "NO_STORED_ANCESTOR"})
            continue
        ancestor_set = list(f.get("ancestor_resolved") or ())
        k = int(reading["treated"])
        served = int(reading["served"])
        uids = [str(u) for u in entry["eval_uids"]]
        raw = np.asarray(entry["raw_per_view"], dtype=np.float64)
        program = np.asarray(entry["program_per_view"], dtype=np.float64)
        index = {uid: i for i, uid in enumerate(uids)}
        pool = [index[u] for u in ancestor_set if u in index]
        anc_gain = float(np.sum([raw[i] - program[i] for i in pool])) / served \
            if pool else 0.0
        anc_harmed = sum(1 for i in pool if (raw[i] - program[i]) < -MATERIAL)
        if not pool or k > len(pool):
            per_unit.append({"position": row["position"],
                             "status": "NOT_A_SUBSET_OF_THE_ANCESTOR",
                             "ancestor_treated": len(pool), "coverage": k})
            continue
        expected_gain = anc_gain * (k / len(pool)) if pool else 0.0
        expected_harmed = anc_harmed * (k / len(pool)) if pool else 0.0
        passes = 0
        for _ in range(RANDOM_DRAWS):
            take = rng.choice(len(pool), size=k, replace=False) if k else []
            gains = np.zeros(served, dtype=np.float64)
            for j in np.atleast_1d(take):
                i = pool[int(j)]
                gains[i] = raw[i] - program[i]
            reading_draw = {
                "treated": int(k), "served": served,
                "aggregate_gain": float(gains.mean()),
                "harmed_fraction": float((gains < -MATERIAL).mean()),
                "max_single_series_harm": max(0.0, float(-gains.min())),
            }
            if runner.authoritative_gate(reading_draw)["passes"]:
                passes += 1
        per_unit.append({
            "position": row["position"], "status": "READ",
            "coverage": k, "ancestor_treated": len(pool),
            "expected_aggregate_gain": round(expected_gain, 6),
            "actual_candidate_aggregate_gain": reading["aggregate_gain"],
            "candidate_minus_random_expectation": round(
                float(reading["aggregate_gain"]) - expected_gain, 6),
            "expected_harmed_series": round(expected_harmed, 4),
            "actual_harmed_series": reading["harmed_series"],
            "gate_pass_rate_estimate": round(passes / RANDOM_DRAWS, 4),
            "candidate_passes_the_gate": bool(
                (row["faces"][face].get("gate") or {}).get("passes")),
        })
    read = [r for r in per_unit if r.get("status") == "READ"]
    return {
        "face": face,
        "draws_per_unit": RANDOM_DRAWS,
        "seed": RANDOM_SEED,
        "expectation_is_analytic": (
            "gains are additive and an unselected series scores exactly 0, so "
            "E[aggregate] = (k/|A|) x ancestor aggregate"),
        "the_pass_rate_is_not_the_gate_applied_to_the_expectation": (
            "the gate is evaluated inside each draw; putting the expected "
            "reading through the gate would report a probability the "
            "experiment never measured"),
        "per_unit": per_unit,
        "mean_expected_aggregate_gain": (
            round(statistics.fmean(
                [r["expected_aggregate_gain"] for r in read]), 6)
            if read else None),
        "mean_candidate_minus_random": (
            round(statistics.fmean(
                [r["candidate_minus_random_expectation"] for r in read]), 6)
            if read else None),
        "mean_gate_pass_rate_of_a_random_retention": (
            round(statistics.fmean(
                [r["gate_pass_rate_estimate"] for r in read]), 4)
            if read else None),
    }


# ---------------------------------------------------------------------------
# the three oracle layers
# ---------------------------------------------------------------------------

def _oracle_l1(population, store: Mapping[str, Any],
               program_labels: Sequence[str], face: str, side: str
               ) -> dict[str, Any]:
    """Per-series, post hoc.  A **relaxed upper bound**, not a Scope.

    With one program in ``program_labels`` this is the treat / do-not-treat
    oracle the Scope line needs.  With several it also picks, per series, which
    program to run -- the Workflow line's version of the same relaxation.
    """
    per_unit, gains, gains_floor = [], [], []
    for row in population:
        if row["side"] != side:
            continue
        entries = {label: store.get(_entry_key(label, row["position"], face))
                   for label in program_labels}
        usable = {label: entry for label, entry in entries.items()
                  if entry is not None and entry.get("verifier_passed")}
        if not usable:
            per_unit.append({"position": row["position"],
                             "status": "NOT_READABLE"})
            gains.append(0.0)
            gains_floor.append(0.0)
            continue
        first = next(iter(usable.values()))
        uids = [str(u) for u in first["eval_uids"]]
        raw = np.asarray(first["raw_per_view"], dtype=np.float64)
        best = np.zeros(len(uids), dtype=np.float64)
        picked_program: dict[str, str] = {}
        for label, entry in sorted(usable.items()):
            program = np.asarray(entry["program_per_view"], dtype=np.float64)
            degenerate = set(entry.get("degenerate_uids") or ())
            per_series = raw - program
            for i, uid in enumerate(uids):
                if uid in degenerate:
                    continue
                if per_series[i] > best[i]:
                    best[i] = float(per_series[i])
                    picked_program[uid] = label
        take = [i for i, uid in enumerate(uids) if best[i] > 0.0]
        aggregate = float(best.mean())
        reading = {"treated": len(take), "served": len(uids),
                   "aggregate_gain": aggregate, "harmed_fraction": 0.0,
                   "max_single_series_harm": 0.0}
        gate = runner.authoritative_gate(reading)
        floor_ok = len(take) >= FLOOR
        gains.append(aggregate)
        gains_floor.append(aggregate if floor_ok else 0.0)
        per_unit.append({"position": row["position"], "status": "READ",
                         "treated": len(take), "served": len(uids),
                         "aggregate_gain": round(aggregate, 6),
                         "harmed_series": 0,
                         "programs_used": sorted(set(picked_program.values())),
                         "gate": gate["passes"],
                         "failed_lines": gate["failed_lines"]})
    return {
        "layer": "L1",
        "programs_in_the_relaxation": list(program_labels),
        "what_it_is": ("per-series post-hoc choice: treat or not, and when "
                       "more than one program is in the space, which one"),
        "action_freedom": ("RELAXED UPPER BOUND: a per-series choice is not "
                           "expressible as a two-clause serving predicate, so "
                           "L1 bounds the geometry, not the selector"),
        "switches": "per unit and per series",
        "risk_and_coverage": ("harm is zero by construction (only positive "
                              "series are treated); the coverage floor is "
                              "reported and a floor-respecting variant falls "
                              "back to raw"),
        "includes_no_revision_and_raw_fallback": True,
        "population_and_scoring": "the same served population, the same sMASE",
        "per_unit": per_unit,
        "mean_aggregate_gain": (round(statistics.fmean(gains), 6)
                                if gains else None),
        "mean_aggregate_gain_respecting_the_coverage_floor": (
            round(statistics.fmean(gains_floor), 6) if gains_floor else None),
    }


def _oracle_l2_l3(scored: Mapping[str, Sequence[Mapping[str, Any]]],
                  labels: Sequence[str], face: str, side: str,
                  l3_label: str) -> dict[str, Any]:
    """Post-hoc best inside the legal candidate space, fixed and switching."""
    positions = [r["position"] for r in scored[labels[0]]
                 if r["side"] == side]
    by_label = {}
    for label in labels:
        rows = {r["position"]: r for r in scored[label] if r["side"] == side}
        by_label[label] = {
            p: _executable_gain(rows[p]["faces"].get(face) or {})
            for p in positions}
    fixed = {label: round(statistics.fmean(
        [by_label[label][p] for p in positions]), 6) for label in labels}
    best_fixed = max(fixed, key=lambda k: fixed[k])
    switching, picks = [], {}
    for p in positions:
        winner = max(labels, key=lambda k: by_label[k][p])
        picks[str(p)] = winner
        switching.append(by_label[winner][p])
    l3_value = fixed.get(l3_label)
    return {
        "layer": "L2 and L3",
        "population_and_scoring": "the same served population, the same sMASE",
        "includes_no_revision_and_raw_fallback": (
            "the ancestor is in the space as NO_REVISION, and an illegal pair "
            "on a unit falls back to raw at exactly 0.0"),
        "risk_and_coverage": (
            "the authoritative gate is reported per unit as a constraint "
            "account rather than used to zero a utility; the gate governs "
            "activation, and this package activates nothing"),
        "L2_segment_fixed": {
            "switches": "no -- one candidate for the whole segment",
            "per_candidate_mean": fixed,
            "best": best_fixed,
            "value": fixed[best_fixed],
        },
        "L2_per_unit_switching": {
            "switches": "yes -- the best candidate is re-chosen on every unit",
            "value": round(statistics.fmean(switching), 6) if switching else None,
            "picks": picks,
        },
        "L3_pre_boundary_selector": {
            "switches": "no -- fixed for the segment, chosen before the boundary",
            "label": l3_label,
            "value": l3_value,
        },
        "L2_minus_L3": (round(fixed[best_fixed] - l3_value, 6)
                        if l3_value is not None else None),
        "how_to_read_L2_minus_L3": (
            "it mixes future information, estimation error and change across "
            "windows; it locates where investment could pay, and it is not a "
            "selector error term"),
    }


# ---------------------------------------------------------------------------
# the package
# ---------------------------------------------------------------------------

def _line_report(name: str, scored: Mapping[str, Sequence[Mapping[str, Any]]],
                 labels: Sequence[str], columns: Mapping[str, Any],
                 population, store, program_for: Mapping[str, str],
                 random_for: str | None,
                 l1_programs: Sequence[str] | None = None) -> dict[str, Any]:
    tables = {}
    for label in labels:
        tables[label] = {
            "all_21_development_units": {
                face: _summarise(scored[label], face) for face in FACES},
            "the_16_units_after_the_boundary": {
                face: _summarise(scored[label], face,
                                 side="after_the_boundary") for face in FACES},
            "the_5_units_before_the_boundary": {
                face: _summarise(scored[label], face,
                                 side="before_the_boundary") for face in FACES},
        }
    oracles = {}
    for face in FACES:
        oracles[face] = {
            "L1": _oracle_l1(population, store,
                             l1_programs or [program_for[labels[0]]], face,
                             "after_the_boundary"),
            "L2_L3": _oracle_l2_l3(scored, labels, face, "after_the_boundary",
                                   columns["aligned"]["picked"]
                                   if columns["aligned"]["picked"] in labels
                                   else labels[0]),
        }
    report = {
        "line": name,
        "candidates": list(labels),
        "columns": columns,
        "per_candidate": tables,
        "per_unit": {label: scored[label] for label in labels},
        "oracles_on_the_16_units_after_the_boundary": oracles,
    }
    if random_for and random_for in scored:
        report["random_retention_reference"] = {
            face: _random_reference(
                [r for r in scored[random_for] if r["side"] == "after_the_boundary"],
                face, store, program_for[labels[0]], population)
            for face in FACES}
        report["random_retention_reference"]["candidate"] = random_for
    return report


def _headline(line: Mapping[str, Any]) -> dict[str, Any]:
    """The numbers the three questions are answered from.  Facts, not verdicts."""
    parent = line["candidates"][0]
    after = {label: {face: line["per_candidate"][label][
        "the_16_units_after_the_boundary"][face] for face in FACES}
        for label in line["candidates"]}
    oracles = line["oracles_on_the_16_units_after_the_boundary"]
    return {
        "parent_label": parent,
        "aligned_selector_picked": line["columns"]["aligned"]["picked"],
        "on_the_16_units_after_the_boundary": {
            label: {
                face: {
                    "mean_aggregate_gain": after[label][face][
                        "mean_executable_aggregate_gain"],
                    "vs_parent": (
                        round(float(after[label][face][
                            "mean_executable_aggregate_gain"])
                            - float(after[parent][face][
                                "mean_executable_aggregate_gain"]), 6)
                        if after[label][face]["mean_executable_aggregate_gain"]
                        is not None else None),
                    "harmed_series": after[label][face]["total_harmed_series"],
                    "worst_single_series_harm": after[label][face][
                        "worst_single_series_harm"],
                    "treated": after[label][face]["total_treated"],
                    "gate_passes_out_of_16": after[label][face]["gate_passes"],
                } for face in FACES}
            for label in line["candidates"]},
        "oracles": {face: {
            "L1_per_series_relaxed_upper_bound": oracles[face]["L1"][
                "mean_aggregate_gain"],
            "L2_segment_fixed_best": oracles[face]["L2_L3"][
                "L2_segment_fixed"]["best"],
            "L2_segment_fixed_value": oracles[face]["L2_L3"][
                "L2_segment_fixed"]["value"],
            "L2_per_unit_switching": oracles[face]["L2_L3"][
                "L2_per_unit_switching"]["value"],
            "L3_value": oracles[face]["L2_L3"]["L3_pre_boundary_selector"][
                "value"],
            "L2_minus_L3": oracles[face]["L2_L3"]["L2_minus_L3"],
        } for face in FACES},
    }


def build(programs: Sequence[str] | None = None,
          out: Path | None = None) -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    preflight_report = smoke.run()
    if not preflight_report["passed"]:
        return {"stage": "M_R0K_SCOPE_WORKFLOW_DEVELOPMENT",
                "status": "BLOCKED_BY_SMOKE",
                "smoke": preflight_report}

    doc = base.load(live.ORDERING)
    state = live._state_at_k1(doc)
    population, excluded = _population(doc)
    store = _load_store()
    cache = runner.ReplayPredictionCache(live.ARM)
    cache.physical_fits = int(sum(int(v.get("physical_fits") or 0)
                                  for v in store.values()))
    fits_at_start = cache.physical_fits

    wanted = list(programs or (["ANCESTOR"] + list(WORKFLOW_CANDIDATES)))
    program_steps = {"ANCESTOR": ANCESTOR_PROGRAM, **WORKFLOW_CANDIDATES}
    builds = []
    for label in wanted:
        builds.append(_build_program(label, program_steps[label], population,
                                     cache, store))

    # ---- the Scope line -------------------------------------------------
    group = next((g for g in outer_loop.census(state["bank"])
                  if g["program_signature"] == "outlier_mad({})"), None)
    bank_rows = list(group["rows"]) if group else []
    space = _enumerate_scope_candidates(bank_rows, state["processed"])
    scope_labels = ["NO_REVISION"]
    scope_scopes = {"NO_REVISION": ANCESTOR_SCOPE}
    for row in space["table"]:
        if row.get("outcome") != "CALIBRATED":
            continue
        label = "%s%s%g" % (row["feature"], row["direction"],
                            row["clause"]["threshold"])
        scope_labels.append(label)
        scope_scopes[label] = row["scope"]
        row["label"] = label

    scope_scored = {label: _policy_rows(label, "ANCESTOR", scope_scopes[label],
                                        population, store, "ANCESTOR")
                    for label in scope_labels}
    scope_pre = {label: _pre_boundary_score(scope_scored[label])
                 for label in scope_labels}

    calibrated_rows = {row["label"]: row for row in space["table"]
                       if row.get("outcome") == "CALIBRATED"}
    original_space, aligned_space, unified_space = [], [], []
    ancestor_pre = float(scope_pre["NO_REVISION"][
        "mean_aggregate_gain_over_the_served_population"])
    for label in scope_labels:
        pre = scope_pre[label]
        coverage = pre["cells_applicable_at_or_above_the_floor"]
        if label == "NO_REVISION":
            row_o = {"label": label, "score": ancestor_pre, "eligible": False,
                     "coverage": coverage,
                     "why_not": "the original rule ranks narrowings only"}
            original_space.append(row_o)
            aligned_space.append({"label": label, "score": ancestor_pre,
                                  "eligible": True, "coverage": coverage})
            unified_space.append({"label": label, "score": ancestor_pre,
                                  "eligible": True, "coverage": coverage})
            continue
        row = calibrated_rows[label]
        eligible_o = row["cells_at_or_above_floor"] > 0
        original_space.append({
            "label": label,
            "score": float(row["pooled_reading_on_treated_rows"][
                "aggregate_gain"]),
            "coverage": row["cells_at_or_above_floor"],
            "eligible": eligible_o,
            "why_not": (None if eligible_o else
                        "reaches no processed cell at or above the floor"),
        })
        aligned_space.append({
            "label": label,
            "score": float(pre["mean_aggregate_gain_over_the_served_population"]),
            "coverage": coverage,
            "eligible": bool(row["preflight"]["accepted"] and eligible_o),
            "why_not": (None if (row["preflight"]["accepted"] and eligible_o)
                        else ("validate_narrowing refuses it"
                              if not row["preflight"]["accepted"] else
                              "no pre-boundary cell reaches the coverage floor,"
                              " so the replay screen returns NOT_APPLICABLE")),
        })
        unified_space.append({
            "label": label,
            "score": float(pre["mean_aggregate_gain_over_the_served_population"]),
            "coverage": coverage,
            "eligible": bool(row["preflight"]["accepted"]
                             and pre["screen_style_pass"]),
            "why_not": (None if (row["preflight"]["accepted"]
                                 and pre["screen_style_pass"])
                        else _why_not_unified(row["preflight"]["accepted"],
                                              pre)),
        })

    scope_columns = {
        "no_revision_or_the_boundary_time_executable_policy": {
            "policy": "the parent version: outlier_mad({}) under z_peak >= 3.0",
            "available": True,
            "why": ("the K0 card is in the active snapshot at every position "
                    "and was never revoked; the Draft's deployable=false is "
                    "the Draft's state, not the ancestor's"),
            "pre_boundary_score": round(ancestor_pre, 6),
        },
        "original_rule": {
            **_select(original_space,
                      rule=("calibrate on the frozen bins, drop the pairs that "
                            "reach no processed cell at or above the coverage "
                            "floor, rank the rest by pooled aggregate gain on "
                            "the TREATED rows -- the M-R0i rule"),
                      ancestor_score=float("-inf")),
            "denominator": "treated bank rows",
            "unfiltered_variant": ("without the coverage filter the objective "
                                   "picks the highest pooled gain outright; "
                                   "M-R0g showed the replay screen then "
                                   "refuses it NOT_APPLICABLE"),
        },
        "aligned": _select(aligned_space,
                           rule=("same candidate space and calibration; rank "
                                 "by aggregate gain over the SERVED population "
                                 "on the five pre-boundary Support faces; "
                                 "accept only at parent + material"),
                           ancestor_score=ancestor_pre),
        "aligned_plus_unified_risk_denominator": {
            **_select(unified_space,
                      rule=("as above, and the risk lines are evaluated on the "
                            "served population instead of the treated rows"),
                      ancestor_score=ancestor_pre),
            "status": ("INDEPENDENT ANALYSIS COLUMN ONLY -- no production rule, "
                       "threshold or gate is changed by it"),
        },
    }
    scope_report = _line_report(
        "scope", scope_scored, scope_labels, scope_columns, population, store,
        {label: "ANCESTOR" for label in scope_labels},
        random_for=(scope_columns["original_rule"]["picked"]
                    if scope_columns["original_rule"]["picked"] in scope_labels
                    else None))
    scope_report["candidate_space"] = space
    scope_report["pre_boundary_evidence"] = scope_pre

    # ---- the Workflow line ----------------------------------------------
    workflow_labels = [label for label in
                       (["ANCESTOR"] + list(WORKFLOW_CANDIDATES))
                       if label in wanted]
    workflow_scored = {label: _policy_rows(label, label, ANCESTOR_SCOPE,
                                           population, store, "ANCESTOR")
                       for label in workflow_labels}
    workflow_pre = {label: _pre_boundary_score(workflow_scored[label])
                    for label in workflow_labels}
    ancestor_pre_w = float(workflow_pre["ANCESTOR"][
        "mean_aggregate_gain_over_the_served_population"])
    aligned_w, unified_w, original_w = [], [], []
    for label in workflow_labels:
        pre = workflow_pre[label]
        legality = next((row for row in
                         preflight_report["checks"][3]["programs"]
                         if row["label"] == label
                         or row["label"].startswith(label)), None)
        legal = bool(legality["legal"]) if legality else True
        pre_readings = [
            ((r["faces"].get("support_face") or {}).get("reading") or {})
            for r in workflow_scored[label]
            if r["side"] == "before_the_boundary"]
        readable = sum(1 for reading in pre_readings
                       if reading.get("status") == "READ")
        # The treated-row denominator: total gain divided by treated series,
        # which is what the threshold tool's ``_reading`` averages over.
        treated_total = sum(int(reading.get("treated") or 0)
                            for reading in pre_readings
                            if reading.get("status") == "READ")
        gain_total = sum(float(reading.get("aggregate_gain") or 0.0)
                         * int(reading.get("served") or 0)
                         for reading in pre_readings
                         if reading.get("status") == "READ")
        treated_rows_score = (gain_total / treated_total if treated_total
                              else 0.0)
        # The incumbent is not re-admitted: it already holds rights, and
        # "keep the parent" is what NO_REVISION means.  Treating it as a
        # candidate that has to clear the admission test would make the two
        # lines answer different questions -- and on this evidence it would
        # rule out the deployed policy itself, which is a finding about the
        # test rather than about the parent.
        incumbent = (label == "ANCESTOR")
        original_w.append({"label": label, "score": treated_rows_score,
                           "coverage": pre[
                               "cells_applicable_at_or_above_the_floor"],
                           "eligible": legal and readable > 0,
                           "why_not": (None if legal else "illegal program")})
        aligned_w.append({
            "label": label,
            "score": float(pre["mean_aggregate_gain_over_the_served_population"]),
            "coverage": pre["cells_applicable_at_or_above_the_floor"],
            "eligible": incumbent or legal,
            "why_not": (None if (incumbent or legal) else "illegal program")})
        unified_w.append({
            "label": label,
            "score": float(pre["mean_aggregate_gain_over_the_served_population"]),
            "coverage": pre["cells_applicable_at_or_above_the_floor"],
            "eligible": incumbent or (legal and pre["screen_style_pass"]),
            "why_not": (None if (incumbent or (legal and pre[
                "screen_style_pass"]))
                        else _why_not_unified(legal, pre)),
            "incumbent": incumbent})

    workflow_columns = {
        "no_revision_or_the_boundary_time_executable_policy": {
            "policy": "the parent program outlier_mad({}) under z_peak >= 3.0",
            "available": True,
            "pre_boundary_score": round(ancestor_pre_w, 6),
        },
        "original_rule": {
            "status": "NOT_APPLICABLE",
            "why": ("the production chain has no Workflow-revision rule at "
                    "all: the outer loop revises Scope only, which is the gap "
                    "the plan's 4.3 proposes closing.  The nearest analogue -- "
                    "rank the programs by their pre-boundary Support gain on "
                    "the treated rows -- is reported as an analogue and is not "
                    "a production rule"),
            "analogue": _select(original_w,
                                rule=("analogue only: rank by pre-boundary "
                                      "Support gain per treated series"),
                                ancestor_score=float("-inf")),
        },
        "aligned": _select(aligned_w,
                           rule=("rank by aggregate gain over the SERVED "
                                 "population on the five pre-boundary Support "
                                 "faces; accept only at parent + material"),
                           ancestor_score=ancestor_pre_w),
        "aligned_plus_unified_risk_denominator": {
            **_select(unified_w,
                      rule=("as above, and the risk lines are evaluated on the "
                            "served population"),
                      ancestor_score=ancestor_pre_w),
            "status": "INDEPENDENT ANALYSIS COLUMN ONLY",
        },
        "time_discipline": (
            "these three candidates were named after the course was read.  "
            "Everything below is a DEVELOPMENT SELECTION REPLAY: it measures "
            "the space and what a selector would do inside it.  It is not an "
            "autonomous proposal by k1, and three failures here do not close "
            "the Workflow direction"),
    }
    workflow_report = _line_report(
        "workflow", workflow_scored, workflow_labels, workflow_columns,
        population, store, {label: label for label in workflow_labels},
        random_for=None, l1_programs=workflow_labels)
    workflow_report["pre_boundary_evidence"] = workflow_pre
    workflow_report["coverage_is_held_fixed"] = (
        "the Scope is the ancestor's on every candidate, so the same-coverage "
        "random-retention reference is degenerate here; treated counts are "
        "reported per candidate instead, and they differ only where a program "
        "is illegal on a face")

    return {
        "stage": "M_R0K_SCOPE_WORKFLOW_DEVELOPMENT",
        "package": "execution plan 4.5, first package",
        "what_this_is_not": [
            "not the old appendix-D full M-W",
            "not a formal evolution experiment",
            "development-level only: every unit here is already exposed",
        ],
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "smoke": preflight_report,
        "fixed_geometry": {
            "ordering": live.ORDERING, "arm": live.ARM,
            "ancestor_scope": ANCESTOR_SCOPE,
            "ancestor_program": [{"op": op, "params": dict(params)}
                                 for op, params in ANCESTOR_PROGRAM],
            "boundary": {"k_index": live.K_INDEX, "position": BOUNDARY,
                         "units_before": 5, "units_after": 16},
            "population": {"development_units": len(population),
                           "excluded": excluded},
            "faces": ["support (the unit's origin)", "delayed (origin + 48)"],
            "not_read": ["the +144 evaluation face", "the 5 TARGET_HELD_IN "
                         "units", "sealed data"],
            "scoring": ("the full served population on every unit; an "
                        "out-of-scope series carries the raw prediction and "
                        "scores exactly 0"),
        },
        "instrument_checks": {
            "raw_pipeline_is_program_independent": _raw_consistency(
                store, wanted),
            "bank_agrees_with_the_store": _bank_agrees_with_the_store(
                bank_rows, population, store),
            "stored_re_mask_equals_the_production_cache": (
                "asserted on every entry at build time; a disagreement raises "
                "InstrumentFault and stops the affected fits"),
        },
        "program_builds": builds,
        "headline": {
            "scope": _headline(scope_report),
            "workflow": _headline(workflow_report),
            "how_to_read_it": (
                "utility is the mean aggregate gain over the served population "
                "on the sixteen units after the selection boundary; the five "
                "pre-boundary units are the selector's evidence and are "
                "reported separately, so a degradation they caused is never "
                "charged to the later sixteen.  The delayed face is the "
                "authoritative one; the Support face is evidence and carries "
                "no gate of its own"),
        },
        "scope_line": scope_report,
        "workflow_line": workflow_report,
        "actual_cost": {
            "llm_calls": 0,
            "physical_consumer_fits": cache.physical_fits,
            "fits_carried_in_from_the_store": fits_at_start,
            "ceiling": MAX_FITS,
            "within_ceiling": cache.physical_fits <= MAX_FITS,
            "counting": ("every physical fit, including smoke, retries and "
                         "fits spent inside a call that then raised; no "
                         "allowance rolled over from an earlier package"),
        },
        "boundary": {
            "skills_activated": 0, "deployment_rights_issued": 0,
            "stores_written": 0, "snapshots_minted": 0,
            "lifecycle_counters_moved": 0, "candidates_changed": 0,
            "thresholds_changed": 0, "operators_added": 0,
            "evaluation_face_reads": 0, "sealed_reads": 0,
            "target_held_in_reads": 0, "llm_calls": 0,
            "gates_analysed_but_no_life_cycle_moved": True,
        },
        "code_state": base.version_check(),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    label = argv[argv.index("--label") + 1] if "--label" in argv else "run1"
    programs = (argv[argv.index("--programs") + 1].split(",")
                if "--programs" in argv else None)
    out = ART / ("m_r0k_scope_workflow_development__%s.json" % label)
    if out.exists() and "--allow-overwrite" not in argv:
        print("refusing to overwrite %s" % out)
        return 2
    report = build(programs=programs, out=out)
    ART.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drafts._plain(report), indent=1,
                              ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("wrote %s" % out)
    print(json.dumps({"cost": report.get("actual_cost"),
                      "scope_picks": {
                          k: (v.get("picked") if isinstance(v, dict) else v)
                          for k, v in (report.get("scope_line") or {}).get(
                              "columns", {}).items()},
                      "workflow_picks": {
                          k: (v.get("picked") if isinstance(v, dict) else v)
                          for k, v in (report.get("workflow_line") or {}).get(
                              "columns", {}).items()}},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
