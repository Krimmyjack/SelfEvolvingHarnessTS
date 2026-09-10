"""The deterministic control again, with one factor changed: the selection rule.

M-R0g ran ``best_stump``'s own pick -- the largest pooled aggregate gain -- and
the real replay screen refused it ``NOT_APPLICABLE``: none of the five
already-processed cells resolved at or above the coverage floor.  M-R0h then
checked the whole finite space at zero cost and found that four of the five
calibrated-feasible candidates *do* reach an applicable cell; the one the
objective picked was the only one that does not.

So this run keeps the objective and adds the filter in front of it:

    among the calibrated-feasible (feature, direction) pairs, drop the ones
    whose calibrated clause reaches no already-processed cell at or above the
    coverage floor, then rank what is left by the same objective and take the
    top one.

The filter uses **only k=1 evidence**: the same 100 bank rows the census
produced, and the five processed cells' own resolution at their own origins --
which is the face ``outer_loop.screen`` replays on.  No later unit is consulted
to choose the candidate, and no truth is read to choose it: Scope resolution
reads deployment-visible feature cards and fits nothing.

Once the candidate is locked it walks the unchanged path: ``clause_from_slow``
calibration on frozen bin edges, ``validate_narrowing`` preflight, the real
replay screen.  If it survives, the verification unit is the **first later cell
in course order** whose delayed face resolves at or above the floor *and* whose
resolved set differs from the ancestor's -- because a cell where the two
predicates select the same series can only speak about the program, never about
the narrowing.  M-R0h says that cell is position 5, ``[0:40]@2856``; the runner
derives it rather than trusting that, and records both.

Every per-cell replay reading is kept, including the ones the screen marks not
applicable.  ``outer_loop.screen`` computes ``treated``, ``aggregate_gain``,
``harmed_fraction`` and ``max_single_series_harm`` for every cell and then
returns only the refusals, so a reading the run already paid fits for is
discarded by the admission verdict.  Nothing new is built for this: the replay
callable is wrapped where it already returns those cells, and the list is put in
the receipt.

**Ceiling: 20 Consumer fits, 0 LLM.**  The screen is 2 fits per processed cell
(5 cells = 10); the verification is 2 per face on one later cell (2 faces = 4).
No model is called: no backend, transport or credential is constructed here.

**This grants nothing**: no Skill activated, no deployment right issued, no
Store written, the +144 evaluation face never read, the course not resumed.  A
failure is not chased with a second candidate.

Run:  python -m evaluation.main_protocol_p4.run_m_r0i_eligible_deterministic_control
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import audit_m_r0b_revision_opportunity as opp
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import scope_threshold_tool as tool

ART = base.ART
MAX_FITS = 20
MAX_LLM = 0
FLOOR = int(contract.RISK["min_treated"])
DELAYED = int(runner.DELAYED_OFFSET)

#: What M-R0h computed the verification cell would be.  Recorded as an
#: expectation and compared against what this run derives, so a silent drift
#: onto a different unit is visible rather than assumed away.
EXPECTED_VERIFY = {"position": 5, "block": "[0:40]", "origin": 2856}


class FitCeiling(RuntimeError):
    """The 20-fit ceiling would be crossed.  Nothing is spent."""


class _EligibleDeterministicProposer:
    """The same objective, behind a coverage filter.  Zero LLM calls.

    ``processed`` are the arm's own already-processed unit contexts; their
    resolution at their own origins is the screen's applicability test, so
    applying it here is the filter asking the question the screen will ask.
    """

    def __init__(self, processed: Sequence[Any]) -> None:
        self.processed = list(processed)
        self.calls: list[dict[str, Any]] = []
        self.table: list[dict[str, Any]] = []

    def _coverage(self, scope: Mapping[str, Any]) -> dict[str, Any]:
        cells = []
        for ctx in self.processed:
            try:
                treated = len(ctx.resolve(scope, ctx.origin))
            except Exception as exc:  # noqa: BLE001 - recorded, not guessed
                cells.append({"unit": ctx.unit, "unresolvable": str(exc)[:160]})
                continue
            cells.append({"unit": ctx.unit, "origin": ctx.origin,
                          "treated": treated,
                          "at_or_above_floor": treated >= FLOOR})
        return {"cells": cells,
                "cells_at_or_above_floor": sum(
                    1 for c in cells if c.get("at_or_above_floor")),
                "eligible": any(c.get("at_or_above_floor") for c in cells)}

    def __call__(self, *, candidate: Mapping[str, Any],
                 rejected: Sequence[Mapping[str, Any]]
                 ) -> Mapping[str, Any] | None:
        rows = list(candidate.get("rows") or ())
        existing = list(dict(candidate.get("base_scope") or {}).get("predicate")
                        or ())
        refused = {(str(r.get("feature")), str(r.get("direction")))
                   for r in (rejected or ())}
        surviving = tool._existing_survivors(rows, existing)

        pairs: set[tuple[str, str]] = set()
        for feature in contract.SCOPE_CLASS["vocabulary"]:
            try:
                edges = tool._edges_for(str(feature), None)
            except Exception:  # noqa: BLE001 - not numeric here
                continue
            for direction in tool.DIRECTIONS:
                for edge in edges:
                    reading = tool._reading(
                        tool._select(surviving, str(feature), direction, edge),
                        tool.BOUNDED_RISK_V1)
                    if reading["feasible"]:
                        pairs.add((str(feature), direction))
                        break

        table: list[dict[str, Any]] = []
        for feature, direction in sorted(pairs):
            try:
                chosen = tool.calibrate(feature=feature, direction=direction,
                                        rows=rows, existing_clauses=existing)
            except tool.NoFeasibleThreshold:
                table.append({"feature": feature, "direction": direction,
                              "status": "CALIBRATE_REFUSES_THE_PAIR"})
                continue
            clause = dict(chosen["clause"])
            scope = {"scope_type": "serving_series_predicate",
                     "predicate": [dict(c) for c in existing] + [clause]}
            coverage = self._coverage(scope)
            table.append({
                "feature": feature, "direction": direction, "clause": clause,
                "objective_pooled_aggregate_gain":
                    chosen["reading"]["aggregate_gain"],
                "pooled_treated": chosen["treated"],
                "pooled_harmed_fraction": chosen["reading"]["harmed_fraction"],
                "historical_replay_coverage": coverage,
                "passes_the_filter": coverage["eligible"],
                "already_refused_this_step": (feature, direction) in refused,
            })
        self.table = table

        pool = [row for row in table
                if row.get("passes_the_filter")
                and not row.get("already_refused_this_step")]
        # The original objective, unchanged: largest aggregate gain, ties to
        # the wider clause and then the smaller threshold, exactly the order
        # ``best_stump`` uses.
        pool.sort(key=lambda row: (-float(row["objective_pooled_aggregate_gain"]),
                                   -int(row["pooled_treated"]),
                                   abs(float(row["clause"]["threshold"]))))
        picked = pool[0] if pool else None
        self.calls.append({
            "candidate": candidate.get("kind"),
            "proposer": "best_stump objective behind a coverage filter",
            "llm_calls": 0,
            "feasible_pairs": len(pairs),
            "passed_the_filter": len(pool),
            "filtered_out": [row["clause"] for row in table
                             if row.get("clause")
                             and not row.get("passes_the_filter")],
            "picked": (dict(picked["clause"]) if picked else None),
            "why": ("the highest-objective candidate that reaches at least one "
                    "already-processed cell at or above the coverage floor"
                    if picked else
                    "no calibrated-feasible candidate survives the filter"),
        })
        if picked is None:
            return None
        return {"scope_clause": {"feature": picked["clause"]["feature"],
                                 "op": picked["clause"]["op"],
                                 "threshold": picked["clause"]["threshold"]}}


def _recording_replay(processed, ledgers: runner.Ledgers,
                      cache: runner.ReplayPredictionCache,
                      sink: list[dict[str, Any]]):
    """The real screen, with the per-cell readings kept instead of dropped."""
    inner = runner.replay_screen_for(processed, ledgers, cache)

    def replay(*, steps, scope):
        estimate = int(getattr(inner, "estimated_fits_per_candidate", 0) or 0)
        if cache.physical_fits + estimate > MAX_FITS:
            raise FitCeiling(
                "the screen's worst case (%d) would take the package past %d "
                "fits (already spent %d)"
                % (estimate, MAX_FITS, cache.physical_fits))
        outcome = inner(steps=steps, scope=scope)
        # Every cell as the screen saw it, before ``screen`` decides which of
        # them it will keep.  ``_applicable`` and ``_violations`` are re-run
        # here only to label each row with the verdict it will receive; the
        # numbers themselves are the ones the fits already bought.
        for cell in (outcome.get("cells") or ()):
            applicable = outer_loop._applicable(cell, min_treated=FLOOR)
            sink.append({
                **{k: v for k, v in dict(cell).items()},
                "applicable": applicable,
                "verdict": ("APPLICABLE" if applicable else "NOT_APPLICABLE"),
                "why_not_applicable": (
                    None if applicable else
                    (cell.get("unusable")
                     or "predicate resolves below the coverage floor")),
                "violations_if_applicable": (
                    outer_loop._violations(
                        cell, material=tool.BOUNDED_RISK_V1.material,
                        max_harmed=tool.BOUNDED_RISK_V1.max_harmed_fraction,
                        max_harm=tool.BOUNDED_RISK_V1.max_single_series_harm)
                    if applicable else None),
                "kept_because": ("outer_loop.screen returns only refusals; "
                                 "this reading was paid for and would "
                                 "otherwise be discarded by the verdict"),
            })
        return outcome

    replay.estimated_fits_per_candidate = getattr(
        inner, "estimated_fits_per_candidate", 0)
    return replay


def _read(cache: runner.ReplayPredictionCache, ctx, origin: int, steps,
          resolved) -> dict[str, Any]:
    key = cache.key(ctx.unit, origin, runner.forecast_p4._config(origin),
                    steps or ())
    if key not in cache._entries and cache.physical_fits + 2 > MAX_FITS:
        raise FitCeiling("a new (unit, face, program) entry would take the "
                         "package past %d fits (spent %d)"
                         % (MAX_FITS, cache.physical_fits))
    return cache.reading(ctx, origin, steps, resolved)


def _scored(reading: Mapping[str, Any], *, label: str, scope) -> dict[str, Any]:
    return {
        "condition": label,
        "scope": dict(scope) if scope else None,
        "treated": reading["treated"],
        "served": reading["served"],
        "coverage": (round(reading["treated"] / reading["served"], 4)
                     if reading["served"] else None),
        "aggregate_gain_over_the_full_population": reading["aggregate_gain"],
        "harmed_fraction_over_the_full_population": reading["harmed_fraction"],
        "max_single_series_harm": reading["max_single_series_harm"],
        "mean_smase": round(float(reading["mean_smase"]), 6),
        "raw_mean_smase": round(float(reading["static_mean_smase"]), 6),
        "improvement_over_raw": round(
            float(reading["static_mean_smase"]) - float(reading["mean_smase"]),
            6),
        "denominator": "every served series; out-of-scope series score as raw",
    }


def _behaviour(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    ga, gb = a["per_series_gain"], b["per_series_gain"]
    uids = sorted(set(ga) | set(gb))
    differing = [uid for uid in uids
                 if repr(float(ga.get(uid, 0.0))) != repr(float(gb.get(uid, 0.0)))]
    return {"series_scored_differently": len(differing),
            "which": differing[:16],
            "treated": {"revised": a["treated"], "ancestor": b["treated"]},
            "identical": not differing}


def _pick_verification_cell(doc, ctxs, scope, ancestor) -> dict[str, Any]:
    """First later cell, in course order, that is readable and contrastive."""
    considered = []
    for cell in base.cells(doc, live.ARM):
        position = int(cell["position"])
        if position <= live.BOUNDARY_POSITION:
            continue
        ctx = ctxs[position]
        origin = ctx.face_origin(DELAYED)
        try:
            got = frozenset(ctx.resolve(scope, origin))
            anc = frozenset(ctx.resolve(ancestor, origin))
        except Exception as exc:  # noqa: BLE001
            considered.append({"position": position,
                               "unresolvable": str(exc)[:160]})
            continue
        row = {"position": position,
               "unit": dict(cell["unit"]),
               "delayed_origin": origin,
               "treated": len(got),
               "ancestor_treated": len(anc),
               "at_or_above_floor": len(got) >= FLOOR,
               "differs_from_ancestor": got != anc}
        considered.append(row)
        if row["at_or_above_floor"] and row["differs_from_ancestor"]:
            return {"found": True, "cell": row, "considered": considered}
    return {"found": False, "cell": None, "considered": considered}


def build() -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    doc = base.load(live.ORDERING)
    state = live._state_at_k1(doc)
    ctxs = {int(c["position"]): opp._unit_ctx(dict(c["unit"]))
            for c in base.cells(doc, live.ARM)}

    ledgers = runner.Ledgers()
    cache = runner.ReplayPredictionCache(live.ARM)
    guard = runner.BudgetGuard(
        ordering_cap=0,
        per_unit_arm_cap=int(contract.PER_UNIT_ARM_BUDGET["llm_calls"]),
        ledgers=ledgers)
    proposer = _EligibleDeterministicProposer(state["processed"])
    per_cell: list[dict[str, Any]] = []
    replay = _recording_replay(state["processed"], ledgers, cache, per_cell)
    budget = outer_loop.OuterBudget(
        outer_llm_per_step=int(contract.OUTER_LLM_PER_STEP),
        replay_fits_remaining=MAX_FITS)

    fault = None
    try:
        record = outer_loop.consolidate(
            bank=state["bank"], ledger=state["ledger"], k_index=live.K_INDEX,
            slow=proposer, replay=replay, budget=budget,
            held_lineage_keys=state["held"],
            bank_boundary={"arm": live.ARM,
                           "units": [c.unit for c in state["processed"]],
                           "excludes": ["the evaluation face (+144)",
                                        "future units", "other arms",
                                        "held-out"]})
        step = record.to_dict()
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        fault = "%s: %s" % (type(exc).__name__, str(exc)[:400])
        step = None

    after = [{"draft_id": d.draft_id, "state": d.state, "revisions": d.revisions,
              "verification_attempts": d.verification_attempts,
              "closed": d.closed, "current_scope": dict(d.current_scope),
              "root_scope": dict(d.root_scope),
              "clauses_added_since_root": d.clauses_added_so_far(),
              "revision_history": [dict(r) for r in d.revision_history],
              "deployable": d.deployable}
             for d in state["ledger"].drafts]
    before = state["drafts_before"]
    target = next((r for r in after if r["draft_id"] == "resupplied_draft_1"),
                  None)
    prior = next((r for r in before if r["draft_id"] == "resupplied_draft_1"),
                 None)
    outcome = next((c.get("outcome") for c in (step or {}).get("candidates")
                    or ()), None)

    steps = [(op, dict(params)) for op, params in
             next(d for d in state["ledger"].drafts
                  if d.draft_id == "resupplied_draft_1").program_steps]
    verification = _verify(doc, ctxs, cache,
                           revised=(target or {}).get("current_scope"),
                           ancestor=(prior or {}).get("current_scope"),
                           steps=steps,
                           screened=(outcome == "DRAFT_REVISED"))

    return {
        "stage": "M_R0I_ELIGIBLE_DETERMINISTIC_CONTROL",
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "the_one_factor": {
            "changed": "the selection rule only",
            "was": "best_stump: rank the feasible triples by pooled aggregate "
                   "gain and take the maximum",
            "is": "drop the calibrated-feasible pairs whose clause reaches no "
                  "already-processed cell at or above the coverage floor, then "
                  "rank what is left by the same objective",
            "unchanged": ["the candidate space and the frozen vocabulary",
                          "the frozen bin edges and clause_from_slow",
                          "validate_narrowing", "the replay screen",
                          "the coverage floor and every risk threshold",
                          "the Draft lifecycle and the information wall"],
            "filter_evidence": "k=1 only: the same 100 bank rows, and the five "
                               "processed cells' resolution at their own "
                               "origins -- the face the screen replays on",
        },
        "authorised_ceiling": {"consumer_fits": MAX_FITS, "llm_calls": MAX_LLM},
        "proposer": {"kind": "deterministic", "llm_calls": 0,
                     "no_backend_constructed": True,
                     "selection_table": proposer.table,
                     "calls": proposer.calls},
        "target": {"ordering": live.ORDERING, "arm": live.ARM,
                   "k_index": live.K_INDEX,
                   "boundary_position": live.BOUNDARY_POSITION,
                   "bank_rows": len(state["bank"]),
                   "units_in_bank": len(state["processed"])},
        "code_state": base.version_check(),
        "outer_step": step,
        "run_fault": fault,
        "replay_cells_every_reading": {
            "why": ("outer_loop.screen returns only cells_not_applicable and "
                    "violations, so an applicable cell that passed leaves no "
                    "reading behind even though its fits were spent"),
            "rows": per_cell,
        },
        "drafts_before": before,
        "drafts_after": after,
        "state_written": {
            "candidate_outcome": outcome,
            "revisions": ((prior or {}).get("revisions"),
                          (target or {}).get("revisions")),
            "scope_before": (prior or {}).get("current_scope"),
            "scope_after": (target or {}).get("current_scope"),
            "clauses_added_since_root": (target or {}).get(
                "clauses_added_since_root"),
            "no_second_shell": len(after) == len(before),
            "still_not_deployable": all(not r["deployable"] for r in after),
            "state": (target or {}).get("state"),
        },
        "verification": verification,
        "actual_cost": {
            "llm_calls": ledgers.llm_total(),
            "consumer_fits_total": cache.physical_fits,
            "of_which_replay_screen": ledgers.replay_fits,
            "of_which_verification": cache.physical_fits - ledgers.replay_fits,
            "within_ceiling": cache.physical_fits <= MAX_FITS,
            "replay_cache": cache.to_dict(),
            "blocked_before_backend": guard.blocked,
        },
        "boundary": {
            "skills_activated": 0, "deployment_rights_issued": 0,
            "stores_written": 0, "snapshots_minted": 0,
            "evaluation_face_reads": 0, "held_out_reads": 0,
            "sealed_reads": 0, "thresholds_changed": 0,
            "coverage_floor_changed": 0, "candidate_space_changed": 0,
            "later_units_scored": (
                1 if verification.get("status") == "RUN" else 0),
        },
        "what_this_is_not": [
            "a deployment: nothing is activated and no right is issued",
            "an LLM comparison: no model is called in this package",
            "a course: the run stops after one later unit",
        ],
    }


def _verify(doc, ctxs, cache, *, revised, ancestor, steps,
            screened: bool) -> dict[str, Any]:
    if not screened:
        return {"status": "NOT_RUN",
                "why": ("the candidate did not survive the screen; a failure "
                        "is not chased with a second candidate"),
                "fits": 0}
    chosen = _pick_verification_cell(doc, ctxs, revised, ancestor)
    if not chosen["found"]:
        return {"status": "NO_ELIGIBLE_LATER_CELL",
                "why": ("no later cell is both at or above the coverage floor "
                        "on its delayed face and behaviourally different from "
                        "the ancestor"),
                "considered": chosen["considered"], "fits": 0}
    picked = chosen["cell"]
    ctx = ctxs[int(picked["position"])]
    before = cache.physical_fits
    out: dict[str, Any] = {
        "status": "RUN",
        "chosen_by": ("first later cell in course order that is at or above "
                      "the coverage floor on its delayed face and differs "
                      "from the ancestor"),
        "cell": picked,
        "matches_the_expectation": {
            "expected": EXPECTED_VERIFY,
            "derived": {"position": int(picked["position"]),
                        "block": picked["unit"]["block"],
                        "origin": int(picked["unit"]["origin"])},
            "same": (int(picked["position"]) == EXPECTED_VERIFY["position"]
                     and int(picked["unit"]["origin"])
                     == EXPECTED_VERIFY["origin"]),
        },
        "cells_considered_before_it": chosen["considered"],
        "served_series": len(ctx.eval_uids),
        "faces": {},
        "note": "the +144 evaluation face is not read here",
    }
    for name, origin in (("support_face", ctx.origin),
                         ("delayed_face", ctx.face_origin(DELAYED))):
        entry: dict[str, Any] = {"origin": origin}
        readings: dict[str, Any] = {}
        for label, scope in (("revised", revised), ("ancestor", ancestor)):
            if not scope:
                continue
            try:
                resolved = frozenset(ctx.resolve(scope, origin))
                reading = _read(cache, ctx, origin, steps, resolved)
            except FitCeiling as exc:
                entry[label] = {"status": "REFUSED_AT_THE_CEILING",
                                "why": str(exc)}
                continue
            except runner.UnitFault as exc:
                entry[label] = {"status": "UNIT_FAULT", "why": str(exc)[:200]}
                continue
            readings[label] = reading
            entry[label] = _scored(reading, label=label, scope=scope)
            entry[label]["gate"] = runner.authoritative_gate(reading)
        try:
            raw = _read(cache, ctx, origin, steps, frozenset())
            entry["raw"] = _scored(raw, label="raw", scope=None)
            entry["raw"]["is_the_reference"] = (
                "identity: no series is treated, so every gain is exactly 0 "
                "and mean_smase is the raw model's own")
        except (FitCeiling, runner.UnitFault) as exc:
            entry["raw"] = {"status": type(exc).__name__, "why": str(exc)[:200]}
        if "revised" in readings and "ancestor" in readings:
            entry["revised_vs_ancestor"] = _behaviour(readings["revised"],
                                                      readings["ancestor"])
            entry["revision_delta_over_the_full_population"] = round(
                readings["revised"]["aggregate_gain"]
                - readings["ancestor"]["aggregate_gain"], 6)
        out["faces"][name] = entry
    out["fits"] = cache.physical_fits - before
    out["pairing"] = ("Support and delayed are the same unit's two faces; the "
                      "delayed face carries the authoritative gate")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    label = argv[argv.index("--label") + 1] if "--label" in argv else "run1"
    report = build()
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / ("m_r0i_eligible_deterministic_control__%s.json" % label)
    if out.exists():
        print("refusing to overwrite %s" % out)
        return 2
    out.write_text(json.dumps(drafts._plain(report), indent=1,
                              ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("wrote %s" % out)
    print(json.dumps({"picked": report["proposer"]["calls"],
                      "state_written": report["state_written"],
                      "cost": report["actual_cost"],
                      "verification_status": report["verification"].get(
                          "status"),
                      "fault": report["run_fault"]},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
