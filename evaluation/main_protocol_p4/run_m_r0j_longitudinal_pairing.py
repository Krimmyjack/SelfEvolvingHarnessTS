"""M-R0i's two Scopes, paired on every remaining legal development unit.

The candidate is not re-chosen and nothing is re-calibrated.  The ancestor and
the revised predicate are frozen exactly as M-R0i left them (and cross-checked
against that receipt), and the only thing this package does is take the paired
reading on each unit the protocol may legally read.

**Population.**  The forward A5-online ordering has 26 cells, all distinct
units.  Five carry ``TARGET_HELD_IN`` and are excluded: they are the Target's
held-in set, not development.  The remaining 21 -- 13 ``SPENT_DEV`` and 8
``UNREAD`` -- are all read, in course order, **without filtering on the
result**.  A unit that turns out to be unevaluable, below the coverage floor,
or behaviourally identical between the two Scopes is reported as such and stays
in the denominator.

**Faces.**  The Support face (the unit's own origin) and the delayed face
(+48), which is where the authoritative gate lives.  The +144 evaluation face
is never read.

**Pairing.**  Both Scopes and raw are re-masks of one pair of fitted models per
(unit, face): the cache key is (arm, unit, face origin, Consumer config, typed
program) and excludes the Scope, so a paired comparison costs what a single
reading costs.  2 fits per (unit, face); 21 units x 2 faces = 84, inside the
authorised 100.

**Scoring** is over the full evaluation population on every unit: out-of-scope
series carry the raw prediction and score a gain of exactly zero, so a
narrowing is scored on the same denominator as its ancestor rather than on its
own.

**Grants nothing**: no Skill activated, no right issued, no Store written, no
lifecycle counter moved, no threshold or candidate touched.  This is a
development reading, not an admission.

Run:  python -m evaluation.main_protocol_p4.run_m_r0j_longitudinal_pairing
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import audit_m_r0b_revision_opportunity as opp
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live

ART = base.ART
MAX_FITS = 100
FLOOR = int(contract.RISK["min_treated"])
DELAYED = int(runner.DELAYED_OFFSET)

#: Exposures this package may read.  ``TARGET_HELD_IN`` is the Target's own
#: held-in set and is excluded by name rather than by any property of the
#: reading, so the exclusion cannot drift with the data.
DEVELOPMENT_EXPOSURES = ("SPENT_DEV", "UNREAD")
EXCLUDED_EXPOSURES = ("TARGET_HELD_IN",)

ANCESTOR = {"scope_type": "serving_series_predicate",
            "predicate": [{"feature": "local_robust_z_peak", "op": ">=",
                           "threshold": 3.0}]}
REVISED = {"scope_type": "serving_series_predicate",
           "predicate": [{"feature": "local_robust_z_peak", "op": ">=",
                          "threshold": 3.0},
                         {"feature": "missing_fraction", "op": ">=",
                          "threshold": 0.05}]}


class FitCeiling(RuntimeError):
    """The 100-fit ceiling would be crossed.  Nothing is spent."""


def _cross_check() -> dict[str, Any]:
    """The two Scopes must be the ones M-R0i actually left behind."""
    path = ART / "m_r0i_eligible_deterministic_control__run1.json"
    if not path.exists():
        return {"checked": False, "why": "no M-R0i receipt on disk"}
    doc = json.loads(path.read_text(encoding="utf-8"))
    written = doc.get("state_written") or {}
    return {
        "checked": True,
        "ancestor_matches": written.get("scope_before") == ANCESTOR,
        "revised_matches": written.get("scope_after") == REVISED,
        "m_r0i_outcome": written.get("candidate_outcome"),
        "m_r0i_revisions": written.get("revisions"),
    }


def _steps(doc: Mapping[str, Any]) -> list[tuple]:
    state = live._state_at_k1(doc)
    draft = next(d for d in state["ledger"].drafts
                 if d.draft_id == "resupplied_draft_1")
    return [(op, dict(params)) for op, params in draft.program_steps]


def _read(cache: runner.ReplayPredictionCache, ctx, origin: int, steps,
          resolved):
    key = cache.key(ctx.unit, origin, runner.forecast_p4._config(origin),
                    steps or ())
    if key not in cache._entries and cache.physical_fits + 2 > MAX_FITS:
        raise FitCeiling("a new (unit, face, program) entry would take the "
                         "package past %d fits (spent %d)"
                         % (MAX_FITS, cache.physical_fits))
    return cache.reading(ctx, origin, steps, resolved)


def _scored(reading: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "treated": reading["treated"],
        "served": reading["served"],
        "coverage": (round(reading["treated"] / reading["served"], 4)
                     if reading["served"] else None),
        "aggregate_gain": reading["aggregate_gain"],
        "harmed_fraction": reading["harmed_fraction"],
        "harmed_series": int(round(float(reading["harmed_fraction"])
                                   * int(reading["served"]))),
        "max_single_series_harm": reading["max_single_series_harm"],
        "mean_smase": round(float(reading["mean_smase"]), 6),
        "raw_mean_smase": round(float(reading["static_mean_smase"]), 6),
    }


def _face(cache, ctx, origin: int, steps) -> dict[str, Any]:
    """One face: coverage for both Scopes, then the paired reading."""
    entry: dict[str, Any] = {"origin": origin}
    resolved: dict[str, Any] = {}
    for label, scope in (("revised", REVISED), ("ancestor", ANCESTOR)):
        try:
            got = frozenset(ctx.resolve(scope, origin))
        except Exception as exc:  # noqa: BLE001 - recorded, not guessed
            entry[label] = {"status": "UNRESOLVABLE", "why": str(exc)[:200]}
            continue
        resolved[label] = got
    entry["coverage"] = {
        label: {"treated": len(got), "served": len(ctx.eval_uids),
                "at_or_above_floor": len(got) >= FLOOR}
        for label, got in resolved.items()}
    if len(resolved) == 2:
        entry["coverage"]["identical_series_set"] = (
            resolved["revised"] == resolved["ancestor"])
        entry["coverage"]["series_only_the_ancestor_treats"] = len(
            resolved["ancestor"] - resolved["revised"])

    readings: dict[str, Any] = {}
    for label in ("revised", "ancestor"):
        if label not in resolved:
            continue
        try:
            reading = _read(cache, ctx, origin, steps, resolved[label])
        except FitCeiling as exc:
            entry[label] = {"status": "REFUSED_AT_THE_CEILING",
                            "why": str(exc)[:200]}
            continue
        except runner.FaceNotEvaluable as exc:
            entry[label] = {"status": "FACE_NOT_EVALUABLE",
                            "why": str(exc)[:200]}
            continue
        except runner.UnitFault as exc:
            entry[label] = {"status": "UNIT_FAULT", "why": str(exc)[:200]}
            continue
        readings[label] = reading
        entry[label] = {**_scored(reading),
                        "gate": runner.authoritative_gate(reading)}
    try:
        raw = _read(cache, ctx, origin, steps, frozenset())
        entry["raw"] = _scored(raw)
    except (FitCeiling, runner.UnitFault) as exc:
        entry["raw"] = {"status": type(exc).__name__, "why": str(exc)[:200]}

    if "revised" in readings and "ancestor" in readings:
        a, b = readings["revised"], readings["ancestor"]
        ga, gb = a["per_series_gain"], b["per_series_gain"]
        uids = sorted(set(ga) | set(gb))
        differing = [u for u in uids
                     if repr(float(ga.get(u, 0.0))) != repr(float(gb.get(u, 0.0)))]
        entry["paired"] = {
            "series_scored_differently": len(differing),
            "behaviour_identical": not differing,
            "revision_delta_aggregate_gain": round(
                a["aggregate_gain"] - b["aggregate_gain"], 6),
            "revision_delta_harmed_fraction": round(
                a["harmed_fraction"] - b["harmed_fraction"], 4),
            "revision_delta_harmed_series": (
                int(round(a["harmed_fraction"] * a["served"]))
                - int(round(b["harmed_fraction"] * b["served"]))),
            "revision_delta_max_single_series_harm": round(
                a["max_single_series_harm"] - b["max_single_series_harm"], 6),
            "revision_delta_treated": a["treated"] - b["treated"],
            "gate": {"revised": runner.authoritative_gate(a)["passes"],
                     "ancestor": runner.authoritative_gate(b)["passes"]},
        }
    else:
        entry["paired"] = {"status": "NOT_PAIRABLE",
                           "why": "one side has no reading on this face"}
    return entry


def _summarise(rows, face: str) -> dict[str, Any]:
    paired = [r["faces"][face]["paired"] for r in rows
              if face in r.get("faces", {})
              and not r["faces"][face]["paired"].get("status")]
    identical = [p for p in paired if p["behaviour_identical"]]
    different = [p for p in paired if not p["behaviour_identical"]]
    deltas = [p["revision_delta_aggregate_gain"] for p in different]
    harm = [p["revision_delta_harmed_series"] for p in different]
    unpairable = [{"position": r["position"], "origin": r["unit"]["origin"],
                   "why": r["faces"][face]["paired"].get("why"),
                   "revised": (r["faces"][face].get("revised") or {}).get(
                       "status"),
                   "ancestor": (r["faces"][face].get("ancestor") or {}).get(
                       "status")}
                  for r in rows if face in r.get("faces", {})
                  and r["faces"][face]["paired"].get("status")]
    return {
        "units_in_the_population": len(rows),
        "units_with_a_paired_reading": len(paired),
        "units_not_pairable": unpairable,
        "behaviour_identical": len(identical),
        "behaviour_different": len(different),
        "identical_carry_no_information_about_the_revision": True,
        "over_the_units_where_behaviour_differs": {
            "n": len(different),
            "aggregate_gain_delta": {
                "mean": (round(statistics.fmean(deltas), 6) if deltas else None),
                "median": (round(statistics.median(deltas), 6)
                           if deltas else None),
                "min": (min(deltas) if deltas else None),
                "max": (max(deltas) if deltas else None),
                "negative": sum(1 for d in deltas if d < 0),
                "positive": sum(1 for d in deltas if d > 0),
                "zero": sum(1 for d in deltas if d == 0),
            },
            "harmed_series_delta": {
                "total": (sum(harm) if harm else 0),
                "reduced_on": sum(1 for h in harm if h < 0),
                "increased_on": sum(1 for h in harm if h > 0),
                "unchanged_on": sum(1 for h in harm if h == 0),
            },
            "gate": {
                "both_pass": sum(1 for p in different
                                 if p["gate"]["revised"] and p["gate"]["ancestor"]),
                "only_revised_passes": sum(
                    1 for p in different
                    if p["gate"]["revised"] and not p["gate"]["ancestor"]),
                "only_ancestor_passes": sum(
                    1 for p in different
                    if p["gate"]["ancestor"] and not p["gate"]["revised"]),
                "neither_passes": sum(
                    1 for p in different
                    if not p["gate"]["revised"] and not p["gate"]["ancestor"]),
            },
        },
        "over_every_paired_unit_including_the_identical_ones": {
            "n": len(paired),
            "mean_aggregate_gain_delta": (
                round(statistics.fmean(
                    [p["revision_delta_aggregate_gain"] for p in paired]), 6)
                if paired else None),
            "why_this_is_the_conservative_number": (
                "a unit where the two Scopes select the same series "
                "contributes an exact zero; including them dilutes the "
                "revision's effect toward zero, excluding them conditions on "
                "the revision having done something"),
        },
    }


def build() -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    doc = base.load(live.ORDERING)
    steps = _steps(doc)
    cache = runner.ReplayPredictionCache(live.ARM)

    population, excluded = [], []
    for cell in base.cells(doc, live.ARM):
        unit = dict(cell["unit"])
        exposure = list(unit.get("exposure") or ())
        row = {"position": int(cell["position"]), "unit": unit,
               "exposure": exposure}
        if any(name in EXCLUDED_EXPOSURES for name in exposure):
            excluded.append({**row, "why": "held for the Target, not development"})
            continue
        if not any(name in DEVELOPMENT_EXPOSURES for name in exposure):
            excluded.append({**row, "why": "exposure is not a development one"})
            continue
        population.append(row)

    rows: list[dict[str, Any]] = []
    for row in population:
        ctx = opp._unit_ctx(row["unit"])
        faces: dict[str, Any] = {}
        for name, origin in (("support_face", ctx.origin),
                             ("delayed_face", ctx.face_origin(DELAYED))):
            faces[name] = _face(cache, ctx, origin, steps)
        rows.append({**row, "served_series": len(ctx.eval_uids),
                     "faces": faces})

    return {
        "stage": "M_R0J_LONGITUDINAL_PAIRING",
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "fixed_from": {
            "artifact": "m_r0i_eligible_deterministic_control__run1.json",
            "ancestor": ANCESTOR,
            "revised": REVISED,
            "cross_check": _cross_check(),
            "nothing_was_reselected": ("the candidate is not re-chosen, the "
                                       "threshold is not re-calibrated and no "
                                       "screen is re-run"),
        },
        "population": {
            "ordering": live.ORDERING, "arm": live.ARM,
            "cells_in_the_ordering": 26,
            "development_units_read": len(population),
            "excluded": excluded,
            "excluded_by": "exposure name, not by any property of the reading",
            "no_result_based_filtering": ("every unit in the population is "
                                          "reported, including unevaluable, "
                                          "below-floor and behaviourally "
                                          "identical ones"),
            "faces": ["the unit's own origin (Support)",
                      "origin + %d (delayed, the authoritative gate)" % DELAYED],
            "not_read": ["the +144 evaluation face", "TARGET_HELD_IN units",
                         "any sealed data"],
        },
        "scoring": {
            "denominator": "every served series on every unit",
            "out_of_scope_series": "carry the raw prediction, gain exactly 0",
            "why": ("a narrowing is scored on the same population as its "
                    "ancestor, not on its own smaller one"),
        },
        "units": rows,
        "summary": {
            "support_face": _summarise(rows, "support_face"),
            "delayed_face": _summarise(rows, "delayed_face"),
        },
        "actual_cost": {
            "llm_calls": 0,
            "consumer_fits_total": cache.physical_fits,
            "within_ceiling": cache.physical_fits <= MAX_FITS,
            "ceiling": MAX_FITS,
            "replay_cache": cache.to_dict(),
            "reuse": ("revised, ancestor and raw are three re-masks of one "
                      "fitted pair per (unit, face); the paired design costs "
                      "what a single reading costs"),
        },
        "boundary": {
            "skills_activated": 0, "deployment_rights_issued": 0,
            "stores_written": 0, "snapshots_minted": 0,
            "lifecycle_counters_moved": 0,
            "candidates_changed": 0, "thresholds_changed": 0,
            "evaluation_face_reads": 0, "sealed_reads": 0,
            "target_held_in_reads": 0,
            "llm_calls": 0,
        },
        "code_state": base.version_check(),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    label = argv[argv.index("--label") + 1] if "--label" in argv else "run1"
    report = build()
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / ("m_r0j_longitudinal_pairing__%s.json" % label)
    if out.exists():
        print("refusing to overwrite %s" % out)
        return 2
    out.write_text(json.dumps(drafts._plain(report), indent=1,
                              ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("wrote %s" % out)
    print(json.dumps({"cross_check": report["fixed_from"]["cross_check"],
                      "units": report["population"]["development_units_read"],
                      "cost": {k: v for k, v in report["actual_cost"].items()
                               if k != "replay_cache"},
                      "summary": report["summary"]},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
