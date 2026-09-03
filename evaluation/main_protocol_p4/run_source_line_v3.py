"""Source line under P4U-v3: ranked refusals, clause-only Slow, two revisions.

v2 established the mechanism and returned a null that could not yet be read.
Three things stood in the way and this run removes exactly those three, without
touching a threshold, an operator, a feature, an endpoint, a cohort or a budget:

* the round's single Slow call is spent on the refusal that is *closest* to
  clearing the four lines, measured as the fewest served series that would have
  to be excluded -- not on whichever candidate Fast happened to propose first;
* Slow writes one Scope clause and the Runtime assembles the manifest, so a
  legal revision can no longer be thrown away for a malformed SHA;
* a Draft that fails the delayed gate is **restricted** rather than destroyed,
  and gets one further bounded narrowing at the next held-in origin.

If a Draft still cannot clear the delayed four lines after its second revision,
this version of the Source line is closed and the null is reported as clean.

What does not cross into the run
--------------------------------
Nothing from the oracle bound (``p4y``): not the feature it selected, not its
threshold, not its ranking, not which series it excluded.  The selection rule
below is arithmetic on the Support per-series vector the refusal already
carries, and its output never reaches the card -- it decides *which* fault Slow
is shown, never *what* to write.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p4 import main_experiment_contract as contract
from evaluation.main_protocol_p4 import main_experiment_contract_v2 as v2
from evaluation.main_protocol_p4 import main_experiment_contract_v3 as v3
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import run_source_line as v1
from evaluation.main_protocol_p4 import run_source_line_v2 as prior
from evaluation.main_protocol_p4 import scope_clause_agent as clause_agent
from evaluation.main_protocol_p4 import scope_narrowing_preflight as narrowing
from evaluation.main_protocol_p4 import scope_repair_distance as distance
from evaluation.main_protocol_p4 import smoke_live_scope as smoke
from evaluation.main_protocol_p4.restricted_draft import _plain
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4w3_source_line_v3.json"
#: The first live v3 run measured the revision mechanism correctly but
#: could not measure a *scoped* Skill: skill_entry_to_dict dropped
#: serving_scope on persistence, so what the gates approved and what the
#: store kept were different objects.  That artifact is kept as it is and
#: this run writes beside it.
OUT_SCOPED = PROJECT_ROOT / (
    "artifacts/main_protocol/p4w3b_source_line_v3_clean_post_fix_replicate_1.json")

#: sol's adjudication: the three v3 runs are separate evidence and must not be
#: read as one another.  Only the third can carry a scientific verdict.
RUN_LABEL = "clean_post_fix_replicate_1"
RUN_LEDGER = {
    "this_run": RUN_LABEL,
    "why_this_one_counts": (
        "the first complete trajectory taken after every instrument fix: the "
        "risk-refusal routing, the clause-only Slow call, the restricted-Draft "
        "lifecycle, and serving_scope persistence"
    ),
    "prior_runs_and_why_they_do_not_count": [
        {"artifact": "p4w2_source_line_v2.card_evidence_missing.json",
         "llm_calls": 33, "status": "SUPERSEDED",
         "why": "the card reaching Slow carried none of the refusal facts"},
        {"artifact": "p4w2_source_line_v2.card_without_applicability.json",
         "llm_calls": 36, "status": "SUPERSEDED_PARTIALLY_VALID",
         "why": "two rounds spent their retry budget on a missing "
                "observable_applicability the manifest schema required"},
        {"artifact": "p4w2_source_line_v2.json",
         "llm_calls": 29, "status": "VALID_FOR_V2",
         "why": "valid, but v2 geometry: one revision, then the Draft was "
                "destroyed"},
        {"artifact": "p4w3_source_line_v3.json",
         "llm_calls": 30, "status": "MECHANISM_ONLY",
         "why": "the lifecycle ran and a Skill was activated, but "
                "skill_entry_to_dict dropped serving_scope on persistence, so "
                "what the gates approved and what the store kept differed; it "
                "cannot evidence a reusable scoped Skill"},
        {"artifact": None, "llm_calls": None, "status": "INTERRUPTED",
         "why": "stopped during the fifth origin, 4 of 5 complete, no artifact "
                "written; its store did confirm serving_scope now persists. "
                "Call count unrecorded -- comparable runs had spent 24 by that "
                "point"},
    ],
    "prior_recorded_llm_calls": 33 + 36 + 29 + 30,
}
FACE = "support_a"
DOMAIN = "p4w3-source-line-v3"

MATERIAL = admission_policy.MATERIAL_THRESHOLD
MAX_HARMED = bounded.BOUNDED_MAX_HARMED_FRACTION
MAX_HARM = bounded.BOUNDED_MAX_SINGLE_SERIES_HARM
MIN_TREATED = distance.MIN_TREATED

#: Unchanged from v2 on purpose.  Removing the protocol friction is meant to
#: buy method evidence out of the same spend, not to buy more spend.
SOURCE_BUDGET_V3 = {
    **prior.SOURCE_BUDGET_V2,
    "why_not_raised": (
        "v2 spent roughly half its calls on manifest schema errors; v3 removes "
        "that cost rather than paying for it twice"
    ),
    "v2_spent_across_three_attempts": 98,
}

RUNTIME_APPLICABILITY = dict(prior.RUNTIME_APPLICABILITY)


# ---------------------------------------------------------------------------
# per-origin machinery
# ---------------------------------------------------------------------------

def _context(machinery: Mapping[str, Any], cell: Any, variant: Mapping[str, Any],
             origin: int) -> dict[str, Any]:
    """Everything one origin needs, built the same way for every use.

    The re-encounter reading has to be taken on an origin the revision did not
    shape, which means building that origin's roster, executor and resolver
    from scratch rather than re-using the ones the revision was derived on.
    """
    at = forecast_p4._cell_at(cell, int(origin))
    config = forecast_p4._config(int(origin))
    roster = at.roster(FACE)
    eval_uids = [str(row["series_uid"]) for row in roster if row["role"] == "eval"]
    executor = machinery["ScopeExecutor"](
        roster, at.values, config,
        evaluate_fn=views.forecast_runtime._evaluate,
        max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION)
    features = smoke._feature_cards(variant, eval_uids, int(origin))
    return {
        "origin": int(origin),
        "at": at,
        "config": config,
        "eval_uids": eval_uids,
        "executor": executor,
        "resolve": v1._resolver(variant, eval_uids),
        "features": features,
        "available": sorted({name for uid in eval_uids
                             for name in (features.get(uid) or {})}),
    }


def _card_builder(features: Mapping[str, Mapping[str, float]],
                  eval_uids: Sequence[str]):
    """The Runner's half of the fault report: features, never outcomes.

    Unchanged from v2, including the anonymity: no UID reaches the card, so a
    predicate over series identity cannot be written even in principle.  The
    revision rule text is shorter here only because the manifest is no longer
    Slow's problem.
    """

    def build(_episode: Any) -> dict[str, Any]:
        return {
            "pattern_id": "p4w3-source-line",
            "observable_signature": {"task_kind": "forecast"},
            "observable_applicability": dict(RUNTIME_APPLICABILITY),
            "fault_code": "RISK_REFUSAL",
            "cause_code": v2.RISK_REFUSAL_ROUTE["attributed_cause"],
            "budget": {"max_harmed_fraction": MAX_HARMED,
                       "max_single_series_harm": MAX_HARM,
                       "min_aggregate_gain": MATERIAL,
                       "min_treated": MIN_TREATED},
            "deployment_visible_features": sorted(
                {name for uid in eval_uids
                 for name, value in (features.get(uid) or {}).items()
                 if isinstance(value, (int, float))}),
            "per_series_features": prior._anonymous_evidence(
                features, eval_uids, None),
            "revision_rule": (
                "the program helps on average and damages a few served series "
                "past the budget; the one thing that may be revised is its "
                "serving scope.  Every clause of risk_refusal.serving_scope is "
                "kept by the runtime; you add exactly one more over a "
                "deployment-visible feature."),
        }

    return build


def _preflight(features: Mapping[str, Mapping[str, float]],
               available: Sequence[str], ledger: drafts.DraftLedger):
    """The narrowing gate, with the lifecycle bound wired into it.

    ``root`` is what turns "at most one added clause" from a statement about
    this step into a statement about the Skill that results.  The ledger is the
    only thing that knows whether ``original`` is itself already a revision.
    """

    def check(original, proposed, _origin):
        if not original:
            return narrowing.NarrowingVerdict(
                accepted=False,
                reason="the probe carried no Scope, so there is nothing to narrow",
            ).to_dict()
        return narrowing.validate_narrowing(
            original, proposed, features=features,
            available_features=list(available),
            root=ledger.root_for_scope(original)).to_dict()

    return check


def _gate(executor: Any, steps: Sequence[tuple], read_origin: int,
          scope: Mapping[str, Any] | None, resolve, *, label: str) -> dict[str, Any]:
    """The four declared lines, read on one window, with nothing inferred.

    Used for both the delayed reading and the independent re-encounter, so the
    two are not two different notions of "passed" that happen to share a name.
    """
    resolved = (frozenset(resolve(scope, read_origin))
                if scope and resolve is not None else None)
    reading = (executor.evaluate(tuple(steps), read_origin,
                                 serving_scope=resolved)
               if resolved is not None
               else executor.evaluate(tuple(steps), read_origin))
    profile = prior._risk(getattr(reading, "per_view_gain", ()) or ())
    treated = 0 if resolved is None else len(resolved)
    served = profile["series_read"] or 0
    lines = {
        "coverage_floor": treated >= MIN_TREATED,
        "aggregate": (profile["aggregate_gain"] or 0.0) >= MATERIAL,
        "harmed_fraction": (profile["harmed_fraction"] or 0.0) <= MAX_HARMED,
        "single_series_harm": profile["max_single_series_harm"] <= MAX_HARM,
    }
    return {
        "reading": label,
        "read_origin": int(read_origin),
        "scope_reresolved": resolved is not None,
        "treated": treated,
        "served": served,
        "coverage": round(treated / served, 4) if served else 0.0,
        **profile,
        "lines": lines,
        "passes": all(lines.values()),
        "failed_lines": [name for name, ok in lines.items() if not ok],
        "thresholds": {"min_treated": MIN_TREATED, "min_aggregate": MATERIAL,
                       "max_harmed_fraction": MAX_HARMED,
                       "max_single_series_harm": MAX_HARM},
    }


class _MergedScopes(Mapping):
    """The frozen initialiser, plus the predicate each restricted Draft reached.

    A resupplied Draft must be probed under the Scope its first revision
    produced.  Re-running the initialiser on its steps would hand it the
    predicate it has already been shown to fail, and the second clause would
    then buy nothing.
    """

    def __init__(self, base: Mapping[str, Any], ledger: drafts.DraftLedger):
        self._base, self._ledger = base, ledger

    def __getitem__(self, candidate_id: str):
        extra = self._ledger.resupplied_scopes()
        if candidate_id in extra:
            return extra[candidate_id]
        return self._base[candidate_id]

    def __iter__(self):
        seen = list(self._base)
        return iter(seen + [key for key in self._ledger.resupplied_scopes()
                            if key not in seen])

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __bool__(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run(*, dry_run: bool) -> dict[str, Any]:
    started = time.time()
    frozen = {"v1": contract.assert_frozen(), "v2": v2.assert_frozen(),
              "v3": v3.assert_frozen()}
    transport = smoke.transport()
    report: dict[str, Any] = {
        "stage": "P4W3_SOURCE_LINE_V3",
        "run_label": RUN_LABEL,
        "run_ledger": RUN_LEDGER,
        "written_at": datetime.now().astimezone().isoformat(),
        "data_version": contract.DATA_VERSION,
        "contract_v3": v3.to_dict(),
        "contract_frozen": frozen,
        "cohort": "source readable[%d:%d]" % contract.SOURCE_SLICE,
        "face": FACE,
        "origins": list(contract.HELD_IN_ORIGINS),
        "source_budget": SOURCE_BUDGET_V3,
        "transport": transport,
    }
    if not all(item["frozen"] for item in frozen.values()):
        report["status"] = "BLOCKED_ON_CONTRACT"
        report["verdict"] = "CONTRACT_NOT_FROZEN"
        return report
    if dry_run or not transport["ready"]:
        report["status"] = ("DRY_RUN" if transport["ready"]
                            else "BLOCKED_ON_TRANSPORT")
        report["llm_calls"] = 0
        report["verdict"] = ("DRY_RUN_OK" if transport["ready"]
                             else "TRANSPORT_NOT_CONFIGURED")
        return report

    m = v1._machinery()
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (  # noqa: PLC0415
        TTHASlowAgent,
    )
    m["admission_policy"].install_policy(bounded.BOUNDED_POLICY)
    loop = m["online_loop"]
    groups = contract.cohorts()
    cell, variant = baselines._cell(groups["source"])
    series0 = np.asarray(cell.values[cell.support_a[0]], dtype=np.float64)

    store = m["SnapshotStore"](PROJECT_ROOT / ".p4w3_source_store")
    controller = m["EditController"](
        store, surfaces=m["SurfaceRegistry"](), router=m["FaultRouter"]())
    h0 = m["compile_snapshot"](
        PROJECT_ROOT / "methods/ttha/harness/h0", verify_lock=False)
    backend = m["agentic"]._default_backend_factory(
        SOURCE_BUDGET_V3["llm_calls"])
    target = m["agentic"].live_transport()
    core = m["TTHAAgentCore"](
        backend,
        m["LocalPublicToolGateway"](
            series0[:contract.HELD_IN_ORIGINS[0]], task_kind="forecast"),
        model=target["model"], base_url=target["base_url"])
    method = m["TTHAMethod"](m["TTHAFastAgent"](core), h0, ())
    # The aggregate-negative path keeps the historical agent; only the
    # risk-refusal branch writes clauses instead of manifests.
    slow_agent = TTHASlowAgent(core)
    clause_slow = clause_agent.ScopeClauseSlowAgent(core)

    ledger = drafts.DraftLedger()
    origins = [int(value) for value in contract.HELD_IN_ORIGINS]
    rounds: list[dict[str, Any]] = []
    activated: list[str] = []
    supply_instability: list[dict[str, Any]] = []

    for index, origin in enumerate(origins):
        ctx = _context(m, cell, variant, origin)
        eval_uids, features = ctx["eval_uids"], ctx["features"]
        core.tools = m["LocalPublicToolGateway"](
            series0[:origin], task_kind="forecast")
        fast_features = dict(m["extract_public_features"](
            series0[:origin], task_kind="forecast"))
        resupplied = ledger.resupplied_programs()

        try:
            result = loop.run_online_round(
                method, ctx["executor"],
                m["runner"]._a5_request(series0, ctx["at"].values, origin, DOMAIN),
                ctx["at"].values,
                origin=origin,
                slow_agent=slow_agent, controller=controller, store=store,
                card_builder=_card_builder(features, eval_uids),
                round_name="p4w3_source_r%d" % (index + 1),
                budget=contract.PER_ARM_BUDGET["probes"],
                allow_slow=True, allow_group_slow=False,
                domain=DOMAIN, period=int(ctx["config"]["period"]),
                fast_features=fast_features,
                allow_fast_skill=True,
                candidate_scopes=_MergedScopes(
                    v1.InitializerScopes(method), ledger),
                scope_resolver=ctx["resolve"],
                scope_revision_preflight=_preflight(
                    features, ctx["available"], ledger),
                program_supply_verifier=ctx["executor"],
                resupplied_programs=resupplied,
                risk_refusal_selector=distance.selector,
                risk_refusal_slow_agent=clause_slow,
            )
        except Exception as exc:  # noqa: BLE001 - a blocked round is a reading
            rounds.append({"origin": origin,
                           "error": "%s: %s" % (type(exc).__name__, str(exc)[:240]),
                           "llm_calls_so_far": getattr(backend, "calls", None)})
            print("  origin %d BLOCKED %s" % (origin, type(exc).__name__),
                  flush=True)
            break

        trace = getattr(method, "last_trace", None)
        candidate_ids = list(getattr(trace, "candidate_program_steps", {}) or {})
        entry: dict[str, Any] = {
            "origin": origin,
            "served_count": len(eval_uids),
            "candidate_ids": candidate_ids,
            "resupplied_candidate_ids": list(result._resupplied_candidate_ids),
            "retrieved_skill_ids": list(
                getattr(trace, "retrieved_skill_ids", ()) or ()),
            "probes": [dict(probe) for probe in result.actual_probed_programs],
            "harm_count": result.harm_count,
            "risk_refusal_count": result.risk_refusal_count,
            "risk_refusals": [dict(row) for row in result.risk_refusals],
            # The ranking, not only the pick: an artifact that recorded the
            # winner alone could not answer whether the rule changed anything.
            "risk_refusal_ranking": distance.select_risk_refusal(
                result.risk_refusals),
            "risk_refusal_selection": result._risk_refusal_selection,
            "slow_trigger": result._slow_trigger,
            "slow_event": result._slow_event,
            "scope_revision_preflight": result._scope_revision_preflight,
            "clause_proposals": [dict(row) for row in clause_slow.proposals],
            "winner_program": result.winner_program,
            # Frozen mappings render as repr strings under json default=str,
            # which made the v3 artifact's winner scopes unreadable by machine.
            "winner_serving_scope": _plain(result._winner_serving_scope),
            "winner_resolved_serving_series": (
                sorted(result._winner_resolved_series)
                if result._winner_resolved_series else None),
            "winner_scope_revision": result._winner_scope_revision,
            "fast_skill_event": result._fast_skill_event,
            "llm_calls_so_far": getattr(backend, "calls", None),
        }
        clause_slow.proposals = []

        # Recorded, never repaired this round: a round with no candidate at all
        # is a different first-fault family from a Scope/Risk conflict, and
        # fixing both at once would make the outcome unattributable to either.
        if not candidate_ids:
            # The note has to describe *this* round.  Pasting the contract's
            # prose in verbatim attached a sentence about origin 2856 to a
            # zero-candidate event at 2136, which is a false record even though
            # the phenomenon is the same one.
            supply_instability.append({
                "origin": origin,
                "candidates_proposed": 0,
                "finding": v3.SEPARATE_FINDING["name"],
                "note": ("the Fast agent proposed no candidate at origin %d; "
                         "recorded as candidate-supply instability and not "
                         "repaired this round, so it cannot be confused with "
                         "the Scope/Risk conflict under test" % origin),
                "first_seen": v3.SEPARATE_FINDING["observed"],
            })

        entry.update(_promote(
            m=m, cell=cell, variant=variant, loop=loop, store=store,
            ledger=ledger, ctx=ctx, result=result, origin=origin,
            origins=origins, index=index, activated=activated))
        rounds.append(entry)
        print("  origin %-5d refusals=%d sel=%-22s slow=%-22s delayed=%s "
              "reenc=%s activated=%s" % (
                  origin, result.risk_refusal_count,
                  (result._risk_refusal_selection or {}).get(
                      "selected_candidate_id"),
                  (result._slow_event or {}).get("stage"),
                  (entry.get("delayed_gate") or {}).get("passes"),
                  (entry.get("re_encounter_gate") or {}).get("passes"),
                  entry.get("activated")), flush=True)

    for draft in ledger.open_drafts():
        # A Draft first restricted at the last held-in origin has no next
        # origin to be revised at.  That is a limit of the geometry, and it is
        # recorded as one rather than counted as a second failure.
        ledger.close(draft, "out_of_origins")

    observed = [
        {"origin": entry["origin"],
         "retrieved": [sid for sid in entry.get("retrieved_skill_ids", ())
                       if sid in activated],
         "deployed_scope": entry.get("winner_serving_scope")}
        for entry in rounds if entry.get("retrieved_skill_ids")]
    survived = [row for row in observed if row["retrieved"]]

    # A run that did not reach every origin has no scientific verdict and must
    # not be able to look like one.  The first attempt at this replicate died
    # on a Windows directory-rename race at the first origin, after six LLM
    # calls, and still wrote ``status: COMPLETE`` with ``A5_TREATMENT_EMPTY``:
    # a blocked instrument reported as an empty treatment.  That is this line's
    # recurring defect class showing up in the reporting layer -- the verdict
    # described something other than what happened.
    blocked = [r for r in rounds if r.get("error")]
    finished_every_origin = not blocked and len(rounds) == len(origins)

    report.update({
        "status": "COMPLETE" if finished_every_origin else "BLOCKED",
        "rounds": rounds,
        "restricted_drafts": ledger.to_dict(),
        "activated_skill_ids": activated,
        "re_encounters_observed": observed,
        "candidate_supply_instability": supply_instability,
        "llm_calls": getattr(backend, "calls", None),
        "wall_seconds": round(time.time() - started, 1),
        "origins_completed": len(rounds),
        "origins_planned": len(origins),
        "blocked_rounds": [
            {"origin": r.get("origin"), "error": r.get("error")}
            for r in blocked],
        "verdict": (
            "RUN_BLOCKED_NO_VERDICT" if not finished_every_origin else
            "SOURCE_SKILL_SURVIVED" if (activated and survived) else
            "SOURCE_SKILL_ACTIVATED_NO_OBSERVED_REENCOUNTER" if activated else
            "A5_TREATMENT_EMPTY"),
        "why_no_verdict": (
            None if finished_every_origin else
            "the run did not complete every held-in origin, so neither a "
            "survival nor an empty-treatment reading can be taken from it"),
        "stopping_rule_reading": (
            contract.STOPPING_RULES["A5_TREATMENT_EMPTY"]
            if finished_every_origin else None),
        "boundary": {**v3.BOUNDARY, "held_out_reads": 0},
    })
    return report


def _promote(*, m, cell, variant, loop, store, ledger, ctx, result, origin,
             origins, index, activated) -> dict[str, Any]:
    """Delayed gate, then an independent re-encounter, then activation.

    Both readings use the same four lines and the same function, so "passed"
    means one thing.  The re-encounter is taken at the *next held-in origin*,
    on a roster and a resolver built there: a window the revision did not shape
    and did not help produce.
    """
    out: dict[str, Any] = {}
    if not result._winner_steps:
        return {"delayed_gate": None, "re_encounter_gate": None,
                "activated": False}

    steps, scope = result._winner_steps, result._winner_serving_scope
    gate = _gate(ctx["executor"], steps, origin + 48, scope, ctx["resolve"],
                 label="delayed")
    out["delayed_gate"] = gate
    loop.open_delayed(result, ctx["executor"], delayed_origin=origin + 48,
                      store=store, scope_resolver=ctx["resolve"])
    out["delayed_event"] = result._delayed_event
    out["delayed_serving_series"] = (sorted(result.delayed_serving_series)
                                     if result.delayed_serving_series else None)
    out["delayed_scope_reresolved"] = result.delayed_scope_reresolved

    selected = (result._risk_refusal_selection or {}).get(
        "selected_candidate_id") or ""
    # A restricted Draft reaches this point two ways: it was refused again and
    # Slow revised it, or it was simply re-probed and admitted.  Looking it up
    # only by the selected refusal misses the second, which is exactly the path
    # the live run took at origin 2376.
    revised = ledger.by_id(selected) or ledger.by_scope(scope)

    if not gate["passes"]:
        out["activated"] = False
        out["re_encounter_gate"] = None
        out["not_activated_because"] = gate["failed_lines"]
        if scope is None:
            out["restriction"] = "no_scope_to_restrict"
            return out
        if revised is not None:
            # Second revision, still short: this Draft's lifecycle is over and
            # so, if nothing else survives, is this version of the Source line.
            ledger.record_revision(
                revised, origin=origin, new_scope=scope,
                preflight=result._scope_revision_preflight,
                support=result._slow_event)
            revised.delayed_failures.append({
                "origin": int(origin), "delayed_origin": origin + 48,
                "failed_lines": gate["failed_lines"], "reading": gate})
            ledger.close(revised, "second_revision_failed_delayed_gate")
            out["restriction"] = "closed_after_second_revision"
            out["draft_id"] = revised.draft_id
            return out
        original = ((result.risk_refusals or [{}])[
            (result._risk_refusal_selection or {}).get(
                "selected_probe_index", 0)] or {}).get("serving_scope")
        draft = ledger.restrict(
            program_steps=steps, root_scope=original or scope,
            current_scope=scope, origin=origin, delayed_reading=gate,
            support_reading=result._slow_event, revisions=1)
        out["restriction"] = "restricted_for_one_more_revision"
        out["draft_id"] = draft.draft_id
        if index + 1 >= len(origins):
            ledger.close(draft, "out_of_origins")
            out["restriction"] = "restricted_but_no_next_origin"
        return out

    # The delayed reading passed.  It is still the window the revision was
    # taken to, so it is not on its own the evidence that the revision
    # transfers; one independent origin has to agree before anything is
    # activated.
    if index + 1 < len(origins):
        later = _context(m, cell, variant, origins[index + 1])
        reenc = _gate(later["executor"], steps, origins[index + 1], scope,
                      later["resolve"], label="independent_re_encounter")
    else:
        reenc = {"reading": "independent_re_encounter", "passes": False,
                 "unavailable": "no later held-in origin remains",
                 "failed_lines": ["no_independent_origin_available"]}
    out["re_encounter_gate"] = reenc
    if not reenc["passes"]:
        out["activated"] = False
        out["not_activated_because"] = reenc.get("failed_lines")
        if revised is not None:
            ledger.close(revised, "re_encounter_failed")
        return out

    out["activated"] = bool(loop.activate_approved(result, store))
    if revised is not None:
        ledger.close(revised, "promoted_to_skill")
        out["draft_id"] = revised.draft_id
    if out["activated"]:
        for row in v1._skill_rows(result._method._active_snapshot()):
            if row["skill_id"] not in activated:
                activated.append(row["skill_id"])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = run(dry_run=args.dry_run)
    destination = (OUT_SCOPED if not args.dry_run
                   else OUT_SCOPED.with_suffix(".dry_run.json"))
    # Two-way guard: a dry run cannot land on the live path, and a live run
    # cannot overwrite a reading that already exists.  A dry run destroyed the
    # v1 per-series vectors once and they could not be rebuilt from anything.
    if destination.exists() and not args.dry_run:
        raise SystemExit(
            "refusing to overwrite an existing live artifact: %s\n"
            "move or rename it first"
            % destination.relative_to(PROJECT_ROOT).as_posix())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print("transport ready : %s" % report["transport"]["ready"])
    print("verdict         : %s" % report.get("verdict"))
    print("llm calls       : %s" % report.get("llm_calls"))
    print("wrote %s" % destination.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
