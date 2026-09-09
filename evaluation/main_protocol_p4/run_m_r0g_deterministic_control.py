"""Step 3: the deterministic proposer, through the same chain, and verified.

The control ``best_stump`` has always been recorded beside every Slow proposal
and never acted on.  This runs it the other way: the deterministic search picks
one candidate under its own frozen rules, and that candidate then walks the
*identical* production path -- ``clause_from_slow`` calibration on frozen bin
edges, ``validate_narrowing`` preflight, the real replay screen over this arm's
already-processed cells -- and, if it survives, is paired against the original
Scope ancestor and against raw on one later unit's Support and delayed faces.

Why it is routed through ``clause_from_slow``: that is where the threshold is
calibrated and where the shadow contrast is recorded.  Handing the tool a
``{feature, op, threshold}`` payload produced by ``best_stump`` instead of by a
model changes *who named the feature* and nothing else -- the same calibration,
the same frozen edges, the same refusal rules, the same shadow record.  The
model is not called: no backend, no transport, no credentials are constructed
anywhere in this module.

**Ceiling: 20 Consumer fits, 0 LLM calls.**  Every physical fit is counted, and
the cache is the reason the whole chain fits inside it: its key is (arm, unit,
face origin, Consumer config, typed program) and deliberately excludes the
Scope, so the revised predicate, the ancestor predicate and raw are three
re-masks of one pair of fitted models rather than three fits.  The screen costs
2 fits per processed cell (5 cells = 10); the verification costs 2 per face
(2 faces = 4).

**This grants nothing.**  No Skill is activated, no deployment right is issued,
no Store is written, the evaluation face (+144) is never read, and the course is
not resumed.  A Draft revision written here is a Draft revision: still
``REVISABLE``, still not deployable.

Scoring is over the **full evaluation population**: ``per_series_gain`` covers
every served series, and series outside the Scope carry the raw prediction and
therefore a gain of exactly zero.  A narrowing that treats fewer series is
scored on the same denominator as its ancestor, not on its own.

Run:  python -m evaluation.main_protocol_p4.run_m_r0g_deterministic_control
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

#: The authorised ceiling for this package.  Enforced before every build, not
#: audited after it.
MAX_FITS = 20
MAX_LLM = 0

#: The later unit the verification pairs on.  Named, not derived, so the run
#: cannot quietly drift onto a different cell.
VERIFY_POSITION = 6
VERIFY_BLOCK = "[0:40]"
VERIFY_ORIGIN = 3576


class FitCeiling(RuntimeError):
    """The 20-fit ceiling would be crossed.  Nothing is spent."""


class _DeterministicProposer:
    """``best_stump``, in the shape ``_clause_for`` expects from Slow.

    Returns the ScopeFit search's own (feature, direction) as a
    ``scope_clause`` payload.  The threshold it carries is the one the search
    already calibrated on frozen edges; ``clause_from_slow`` ignores whatever
    is in that field and recalibrates, exactly as it does for a model, so the
    number is passed only to keep the payload schema-shaped.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, candidate: Mapping[str, Any],
                 rejected: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        existing = list(dict(candidate.get("base_scope") or {}).get("predicate")
                        or ())
        refused = {(str(row.get("feature")), str(row.get("direction")))
                   for row in (rejected or ())}
        stump = tool.best_stump(rows=candidate.get("rows") or (),
                               existing_clauses=existing)
        self.calls.append({"candidate": candidate.get("kind"),
                           "proposer": "best_stump",
                           "llm_calls": 0,
                           "already_refused": sorted(refused),
                           "outcome": stump.get("outcome"),
                           "feasible_count": stump.get("feasible_count"),
                           "returned": (dict(stump["clause"])
                                        if stump.get("clause") else None)})
        if stump.get("outcome") != "BEST_STUMP":
            return None
        if (str(stump["feature"]), str(stump["direction"])) in refused:
            # The tool already refused this exact pair; the search has no
            # second opinion to offer, so it abstains rather than repeating.
            return None
        return {"scope_clause": {"feature": stump["feature"],
                                 "op": stump["direction"],
                                 "threshold": stump["threshold"]}}


def _capped_replay(processed, ledgers: runner.Ledgers,
                   cache: runner.ReplayPredictionCache):
    inner = runner.replay_screen_for(processed, ledgers, cache)

    def replay(*, steps, scope):
        estimate = int(getattr(inner, "estimated_fits_per_candidate", 0) or 0)
        if cache.physical_fits + estimate > MAX_FITS:
            raise FitCeiling(
                "the screen's worst case (%d) would take the package past %d "
                "fits (already spent %d)"
                % (estimate, MAX_FITS, cache.physical_fits))
        return inner(steps=steps, scope=scope)

    replay.estimated_fits_per_candidate = getattr(
        inner, "estimated_fits_per_candidate", 0)
    return replay


def _read(cache: runner.ReplayPredictionCache, ctx, origin: int, steps,
          resolved) -> dict[str, Any]:
    """One cached reading, refusing rather than crossing the ceiling."""
    key = cache.key(ctx.unit, origin, runner.forecast_p4._config(origin),
                    steps or ())
    if key not in cache._entries and cache.physical_fits + 2 > MAX_FITS:
        raise FitCeiling("a new (unit, face, program) entry would take the "
                         "package past %d fits (spent %d)"
                         % (MAX_FITS, cache.physical_fits))
    return cache.reading(ctx, origin, steps, resolved)


def _scored(reading: Mapping[str, Any], *, label: str,
            scope: Mapping[str, Any] | None) -> dict[str, Any]:
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
    """Do two Scopes actually do anything different on this face?"""
    ga, gb = a["per_series_gain"], b["per_series_gain"]
    uids = sorted(set(ga) | set(gb))
    differing = [uid for uid in uids
                 if repr(float(ga.get(uid, 0.0))) != repr(float(gb.get(uid, 0.0)))]
    return {
        "series_scored_differently": len(differing),
        "which": differing[:12],
        "treated": {"revised": a["treated"], "ancestor": b["treated"]},
        "identical": not differing,
    }


def build() -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    doc = base.load(live.ORDERING)
    state = live._state_at_k1(doc)

    ledgers = runner.Ledgers()
    cache = runner.ReplayPredictionCache(live.ARM)
    guard = runner.BudgetGuard(
        ordering_cap=0,
        per_unit_arm_cap=int(contract.PER_UNIT_ARM_BUDGET["llm_calls"]),
        ledgers=ledgers)
    proposer = _DeterministicProposer()
    replay = _capped_replay(state["processed"], ledgers, cache)
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
              "closed": d.closed,
              "current_scope": dict(d.current_scope),
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

    verification = _verify(doc, cache, ledgers,
                           revised=(target or {}).get("current_scope"),
                           ancestor=(prior or {}).get("current_scope"),
                           steps=_steps(state),
                           screened=(outcome == "DRAFT_REVISED"))

    return {
        "stage": "M_R0G_DETERMINISTIC_CONTROL",
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "what_this_is": (
            "the ScopeFit shadow search, promoted from a recorded contrast to "
            "an actual proposer, through the identical calibration, preflight "
            "and replay path, then paired against its Scope ancestor and raw "
            "on one later unit"),
        "what_this_is_not": [
            "a deployment: no Skill activated, no Store written, no rights "
            "issued",
            "an LLM comparison: n=1 here against n=1 and n=2 there",
            "a course: the run stops after one later unit",
        ],
        "authorised_ceiling": {"consumer_fits": MAX_FITS, "llm_calls": MAX_LLM},
        "proposer": {
            "kind": "deterministic",
            "search": "scope_threshold_tool.best_stump",
            "llm_calls": 0,
            "no_backend_constructed": True,
            "routed_through": ("outer_loop._clause_for -> "
                               "scope_threshold_tool.clause_from_slow, the "
                               "same calibration the model's payload gets"),
            "calls": proposer.calls,
        },
        "target": {"ordering": live.ORDERING, "arm": live.ARM,
                   "k_index": live.K_INDEX,
                   "boundary_position": live.BOUNDARY_POSITION,
                   "bank_rows": len(state["bank"]),
                   "units_in_bank": len(state["processed"])},
        "code_state": base.version_check(),
        "outer_step": step,
        "run_fault": fault,
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
            "of_which_verification": (cache.physical_fits
                                      - ledgers.replay_fits),
            "within_ceiling": cache.physical_fits <= MAX_FITS,
            "replay_cache": cache.to_dict(),
            "blocked_before_backend": guard.blocked,
        },
        "boundary": {
            "skills_activated": 0, "deployment_rights_issued": 0,
            "stores_written": 0, "snapshots_minted": 0,
            "evaluation_face_reads": 0, "held_out_reads": 0,
            "sealed_reads": 0, "thresholds_changed": 0,
            "production_code_changed": 0,
            # Derived, not asserted: the verification is skipped when the
            # candidate does not survive the screen, and a field that always
            # read 1 would claim a later unit was scored when none was.
            "later_units_touched": (
                1 if verification.get("status") == "RUN" else 0),
            "which_later_unit": ({"position": VERIFY_POSITION,
                                  "block": VERIFY_BLOCK,
                                  "origin": VERIFY_ORIGIN}
                                 if verification.get("status") == "RUN"
                                 else None),
        },
    }


def _steps(state: Mapping[str, Any]) -> list[tuple]:
    draft = next(d for d in state["ledger"].drafts
                 if d.draft_id == "resupplied_draft_1")
    return [(op, dict(params)) for op, params in draft.program_steps]


def _verify(doc: Mapping[str, Any], cache: runner.ReplayPredictionCache,
            ledgers: runner.Ledgers, *, revised, ancestor, steps,
            screened: bool) -> dict[str, Any]:
    """Support and delayed, revised vs ancestor vs raw, on the named unit."""
    if not screened:
        return {"status": "NOT_RUN",
                "why": ("the candidate did not survive the screen; the "
                        "authorisation says a failure is not chased with a "
                        "second candidate"),
                "fits": 0}
    cell = next((c for c in base.cells(doc, live.ARM)
                 if int(c["position"]) == VERIFY_POSITION), None)
    if cell is None:
        return {"status": "BLOCKED", "why": "position %d not in the record"
                                            % VERIFY_POSITION, "fits": 0}
    unit = dict(cell["unit"])
    if (str(unit.get("block")) != VERIFY_BLOCK
            or int(unit.get("origin")) != VERIFY_ORIGIN):
        return {"status": "BLOCKED",
                "why": "position %d is %s@%s, not %s@%s"
                       % (VERIFY_POSITION, unit.get("block"),
                          unit.get("origin"), VERIFY_BLOCK, VERIFY_ORIGIN),
                "fits": 0}
    ctx = opp._unit_ctx(unit)
    before = cache.physical_fits
    faces = {"support_face": ctx.origin,
             "delayed_face": ctx.face_origin(runner.DELAYED_OFFSET)}
    out: dict[str, Any] = {"status": "RUN", "unit": ctx.unit,
                           "served_series": len(ctx.eval_uids),
                           "faces": {}, "note": (
                               "the +144 evaluation face is not read here")}
    for name, origin in faces.items():
        entry: dict[str, Any] = {"origin": origin}
        readings: dict[str, Any] = {}
        for label, scope in (("revised", revised), ("ancestor", ancestor)):
            if not scope:
                continue
            try:
                resolved = ctx.resolve(scope, origin)
                reading = _read(cache, ctx, origin, steps, frozenset(resolved))
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
            entry["raw"]["gate"] = runner.authoritative_gate(raw)
        except (FitCeiling, runner.UnitFault) as exc:
            entry["raw"] = {"status": type(exc).__name__, "why": str(exc)[:200]}
        if "revised" in readings and "ancestor" in readings:
            entry["revised_vs_ancestor"] = _behaviour(readings["revised"],
                                                      readings["ancestor"])
            gain = (readings["revised"]["aggregate_gain"]
                    - readings["ancestor"]["aggregate_gain"])
            entry["revision_delta_over_the_full_population"] = round(gain, 6)
        out["faces"][name] = entry
    out["fits"] = cache.physical_fits - before
    out["pairing"] = (
        "Support and delayed are the same unit's two faces; the delayed face "
        "carries the authoritative gate and the Support face does not")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    label = argv[argv.index("--label") + 1] if "--label" in argv else "run1"
    report = build()
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / ("m_r0g_deterministic_control__%s.json" % label)
    if out.exists():
        print("refusing to overwrite %s" % out)
        return 2
    out.write_text(json.dumps(drafts._plain(report), indent=1,
                              ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("wrote %s" % out)
    print(json.dumps({"proposer": report["proposer"]["calls"],
                      "state_written": report["state_written"],
                      "cost": report["actual_cost"],
                      "fault": report["run_fault"]},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
