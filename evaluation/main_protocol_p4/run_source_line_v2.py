"""Source line under P4U-v2: risk refusals routed, Scopes revised, live Slow.

v1 formed zero Skills.  Seven of nine probes cleared the aggregate line and
every one was refused for tail risk, and the refusal reached nothing: the fault
router read only the aggregate gain, so a round of positive-but-refused
candidates recorded zero faults.  v2 routes them.

What is different, and only this:

* a materially positive candidate refused on the tail budget is attributed
  ``RISK_GAP`` and reaches Slow through exactly one surface, the Skill ADD;
* Slow's program is frozen to the probe's own -- the Runtime puts it in the
  card's typed-patch whitelist, which is the only place ``_steps_for_patch_id``
  reads, so no other program can be bound;
* the only thing Slow may write is a monotone narrowing of that probe's Scope,
  and ``scope_narrowing_preflight`` refuses anything else;
* the Support replay is read **under the revised Scope**, because replaying it
  globally reproduces the configuration that was just refused;
* the delayed reading is the gate.  It is taken at origin+48 with the predicate
  **re-resolved from that origin's own features**, and it must clear all four
  declared lines before the Draft is activated.

The Support reading is the feedback the revision was derived from, so it cannot
also be the evidence that the revision generalises.  That is why it is recorded
as adaptation and never as an endpoint.

Evidence the Slow call is given, and evidence it is not
------------------------------------------------------
The card carries the refusal: the program, the current Scope, the refusal
reason, and one row per served series pairing that series' deployment-visible
features with its observed gain.  The rows are **anonymous** -- no UID reaches
the card at all, so a predicate over series identity cannot be written even in
principle, which is a stronger guarantee than refusing it at the Scope grammar.

Nothing from the oracle bound (``p4y``) crosses: not the feature it selected,
not its threshold, not its ranking, not which series it excluded.  That audit
answered one question before this run -- whether the revision class contains a
feasible Scope at all -- so that a failure here is attributable to Slow rather
than to an impossible task.
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
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import run_source_line as v1
from evaluation.main_protocol_p4 import scope_initializer as initializer
from evaluation.main_protocol_p4 import scope_narrowing_preflight as narrowing
from evaluation.main_protocol_p4 import smoke_live_scope as smoke
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4w2_source_line_v2.json"
FACE = "support_a"
DOMAIN = "p4w2-source-line-v2"

MATERIAL = admission_policy.MATERIAL_THRESHOLD
MAX_HARMED = bounded.BOUNDED_MAX_HARMED_FRACTION
MAX_HARM = bounded.BOUNDED_MAX_SINGLE_SERIES_HARM
MIN_TREATED = 5

#: P4U-v2 puts ``observable_applicability`` under the Runtime, not Slow: it
#: decides when the Skill is *retrieved*, which is a different question from
#: which served series it may treat, and only the latter is on trial.  But the
#: manifest schema requires the field, so withholding it does not freeze it --
#: it makes every proposal invalid.  Two rounds of the first valid run burned
#: their whole retry budget on ``missing fields: ['observable_applicability']``
#: and never reached a Scope judgement at all.  So it is handed over to be
#: copied verbatim, which is what "fixed by the Runtime" has to mean in a
#: schema that will not accept its absence.
RUNTIME_APPLICABILITY = {
    "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]
}

#: The Source line is not an arm and does not enter the A3/A5 comparison, so
#: this budget is its own.  v1 spent 28 of 40 on Fast alone; v2 adds one Slow
#: call per routed refusal, so the ceiling is raised rather than the arms'.
SOURCE_BUDGET_V2 = {
    "llm_calls": 60,
    "rounds": len(contract.HELD_IN_ORIGINS),
    "face": FACE,
    "is_an_arm": False,
    "enters_a3_a5_comparison": False,
    "v1_spent": 28,
    "why_raised": "v1's ceiling covered Fast only; routing adds a Slow call",
}


def _anonymous_evidence(features: Mapping[str, Mapping[str, float]],
                        eval_uids: Sequence[str],
                        per_series: Sequence[float] | None) -> list[dict[str, Any]]:
    """One row per served series: its features, its outcome, and no name.

    Withholding the UID is not cosmetic.  A Scope must be a predicate over
    deployment-visible features to transfer at all, and the surest way to
    guarantee that is to make series identity unavailable to the writer rather
    than to refuse it afterwards.
    """
    rows = []
    for index, uid in enumerate(eval_uids):
        card = dict(features.get(uid) or {})
        row = {"series_index": index}
        row.update({name: round(float(value), 6)
                    for name, value in sorted(card.items())
                    if isinstance(value, (int, float))
                    and np.isfinite(float(value))})
        if per_series is not None and index < len(per_series):
            row["observed_gain"] = round(float(per_series[index]), 6)
        rows.append(row)
    return rows


def _card_builder(features: Mapping[str, Mapping[str, float]],
                  eval_uids: Sequence[str]):
    """The half of the fault report the Runner owns: features, never outcomes.

    The refusal facts -- reason, budgets breached, the current Scope and the
    per-series gain vector -- are injected by the Runtime at the moment the
    refusal is routed, because only it knows them then.  An earlier version
    passed them through a holder the Runner filled after the round had already
    returned, so the card that actually reached Slow carried none of them and
    the run measured nothing.

    Both halves are **positional**: row ``i`` here describes the same served
    series as gain ``i`` there, and no UID appears in either.  A predicate over
    series identity therefore cannot be written, rather than being refused
    after it has been.
    """

    def build(_episode: Any) -> dict[str, Any]:
        return {
            # The signature the natural forecast line has always emitted;
            # widening it here would be the Runner choosing how easily its own
            # Skills are retrieved.
            "pattern_id": "p4w2-source-line",
            "observable_signature": {"task_kind": "forecast"},
            "observable_applicability": dict(RUNTIME_APPLICABILITY),
            "fault_code": "RISK_REFUSAL",
            "cause_code": v2.RISK_REFUSAL_ROUTE["attributed_cause"],
            "budget": {"max_harmed_fraction": MAX_HARMED,
                       "max_single_series_harm": MAX_HARM,
                       "min_aggregate_gain": MATERIAL},
            "deployment_visible_features": sorted(
                {name for uid in eval_uids
                 for name, value in (features.get(uid) or {}).items()
                 if isinstance(value, (int, float))}),
            "per_series_features": _anonymous_evidence(
                features, eval_uids, None),
            "revision_rule": (
                "the program helps on average and damages a few served series "
                "past the budget; the one thing that may be revised is its "
                "serving scope.  Keep every clause of "
                "risk_refusal.serving_scope and add at most one more over a "
                "deployment-visible feature, then put the result in the new "
                "Skill's serving_scope field.  A clause naming a series is "
                "refused at construction.  Copy observable_applicability "
                "verbatim into the manifest and into the new Skill: it is "
                "fixed by the Runtime and is not what this fault is about."),
        }

    return build


def _preflight(features: Mapping[str, Mapping[str, float]],
               available: Sequence[str]):
    def check(original, proposed, _origin):
        if not original:
            return narrowing.NarrowingVerdict(
                accepted=False,
                reason="the probe carried no Scope, so there is nothing to narrow",
            ).to_dict()
        return narrowing.validate_narrowing(
            original, proposed, features=features,
            available_features=list(available)).to_dict()

    return check


def _risk(vector: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(list(vector), dtype=np.float64)
    worst = float(-values.min()) if values.size and values.min() < 0 else 0.0
    return {
        "series_read": int(values.size),
        "aggregate_gain": round(float(values.mean()), 6) if values.size else None,
        "harmed_count": int((values < -MATERIAL).sum()),
        "harmed_fraction": round(float((values < -MATERIAL).mean()), 4)
        if values.size else None,
        "max_single_series_harm": round(worst, 6),
        "per_series_gain": [round(float(value), 6) for value in values],
    }


def _delayed_gate(executor: Any, steps: Sequence[tuple], delayed_origin: int,
                  scope: Mapping[str, Any] | None, resolve) -> dict[str, Any]:
    """The endpoint: the revised policy, read on a window it did not shape.

    Taken separately from the method's own delayed stage so the four numbers
    the decision used are in the artifact rather than inferred from a verdict.
    """
    resolved = (frozenset(resolve(scope, delayed_origin))
                if scope and resolve is not None else None)
    reading = (executor.evaluate(tuple(steps), delayed_origin,
                                 serving_scope=resolved)
               if resolved is not None
               else executor.evaluate(tuple(steps), delayed_origin))
    profile = _risk(getattr(reading, "per_view_gain", ()) or ())
    treated = 0 if resolved is None else len(resolved)
    served = profile["series_read"] or 0
    lines = {
        "coverage_floor": treated >= MIN_TREATED,
        "aggregate": (profile["aggregate_gain"] or 0.0) >= MATERIAL,
        "harmed_fraction": (profile["harmed_fraction"] or 0.0) <= MAX_HARMED,
        "single_series_harm": profile["max_single_series_harm"] <= MAX_HARM,
    }
    return {
        "delayed_origin": int(delayed_origin),
        "scope_reresolved": resolved is not None,
        "treated": treated,
        "served": served,
        "coverage": round(treated / served, 4) if served else 0.0,
        **profile,
        "lines": lines,
        "passes": all(lines.values()),
        "thresholds": {"min_treated": MIN_TREATED, "min_aggregate": MATERIAL,
                       "max_harmed_fraction": MAX_HARMED,
                       "max_single_series_harm": MAX_HARM},
    }


def run(*, dry_run: bool) -> dict[str, Any]:
    started = time.time()
    frozen_v1, frozen_v2 = contract.assert_frozen(), v2.assert_frozen()
    transport = smoke.transport()
    report: dict[str, Any] = {
        "stage": "P4W2_SOURCE_LINE_V2",
        "written_at": datetime.now().astimezone().isoformat(),
        "data_version": contract.DATA_VERSION,
        "contract_v2": v2.to_dict(),
        "contract_frozen": {"v1": frozen_v1, "v2": frozen_v2},
        "cohort": "source readable[%d:%d]" % contract.SOURCE_SLICE,
        "face": FACE,
        "origins": list(contract.HELD_IN_ORIGINS),
        "source_budget": SOURCE_BUDGET_V2,
        "transport": transport,
    }
    if not (frozen_v1["frozen"] and frozen_v2["frozen"]):
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

    store = m["SnapshotStore"](PROJECT_ROOT / ".p4w2_source_store")
    controller = m["EditController"](
        store, surfaces=m["SurfaceRegistry"](), router=m["FaultRouter"]())
    h0 = m["compile_snapshot"](
        PROJECT_ROOT / "methods/ttha/harness/h0", verify_lock=False)
    backend = m["agentic"]._default_backend_factory(
        SOURCE_BUDGET_V2["llm_calls"])
    target = m["agentic"].live_transport()
    core = m["TTHAAgentCore"](
        backend,
        m["LocalPublicToolGateway"](
            series0[:contract.HELD_IN_ORIGINS[0]], task_kind="forecast"),
        model=target["model"], base_url=target["base_url"])
    method = m["TTHAMethod"](m["TTHAFastAgent"](core), h0, ())
    slow_agent = TTHASlowAgent(core)

    rounds: list[dict[str, Any]] = []
    activated: list[str] = []
    for index, origin in enumerate(contract.HELD_IN_ORIGINS):
        origin = int(origin)
        at = forecast_p4._cell_at(cell, origin)
        config = forecast_p4._config(origin)
        roster = at.roster(FACE)
        eval_uids = [str(row["series_uid"]) for row in roster
                     if row["role"] == "eval"]
        executor = m["ScopeExecutor"](
            roster, at.values, config,
            evaluate_fn=views.forecast_runtime._evaluate,
            max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION)
        resolve = v1._resolver(variant, eval_uids)
        features = smoke._feature_cards(variant, eval_uids, origin)
        available = sorted({name for uid in eval_uids
                            for name in (features.get(uid) or {})})
        core.tools = m["LocalPublicToolGateway"](
            series0[:origin], task_kind="forecast")
        fast_features = dict(m["extract_public_features"](
            series0[:origin], task_kind="forecast"))

        try:
            result = loop.run_online_round(
                method, executor,
                m["runner"]._a5_request(series0, at.values, origin, DOMAIN),
                at.values,
                origin=origin,
                slow_agent=slow_agent, controller=controller, store=store,
                card_builder=_card_builder(features, eval_uids),
                round_name="p4w2_source_r%d" % (index + 1),
                budget=contract.PER_ARM_BUDGET["probes"],
                allow_slow=True, allow_group_slow=False,
                domain=DOMAIN, period=int(config["period"]),
                fast_features=fast_features,
                allow_fast_skill=True,
                candidate_scopes=v1.InitializerScopes(method),
                scope_resolver=resolve,
                scope_revision_preflight=_preflight(features, available),
                program_supply_verifier=executor,
            )
        except Exception as exc:  # noqa: BLE001 - a blocked round is a reading
            rounds.append({"origin": origin,
                           "error": "%s: %s" % (type(exc).__name__, str(exc)[:240]),
                           "llm_calls_so_far": getattr(backend, "calls", None)})
            print("  origin %d BLOCKED %s" % (origin, type(exc).__name__),
                  flush=True)
            break

        entry: dict[str, Any] = {
            "origin": origin,
            "served_count": len(eval_uids),
            "candidate_ids": list(getattr(
                getattr(method, "last_trace", None), "candidate_program_steps", {})
                or {}),
            "retrieved_skill_ids": list(getattr(
                getattr(method, "last_trace", None), "retrieved_skill_ids", ()) or ()),
            "probes": [dict(probe) for probe in result.actual_probed_programs],
            "harm_count": result.harm_count,
            "risk_refusal_count": result.risk_refusal_count,
            "risk_refusals": [dict(row) for row in result.risk_refusals],
            "slow_trigger": result._slow_trigger,
            "slow_event": result._slow_event,
            "scope_revision_preflight": result._scope_revision_preflight,
            "proposed_scope": (result._slow_event or {}).get("serving_scope"),
            # A Scope proposed on a round that failed before the pending path
            # is never preflighted by the loop, so its legality would go
            # unrecorded.  Recomputed here as a *reading*, never as a gate.
            "proposed_scope_preflight_offline": (
                _preflight(features, available)(
                    ((result.risk_refusals or [{}])[0]).get("serving_scope"),
                    (result._slow_event or {}).get("serving_scope"), origin)
                if ((result._slow_event or {}).get("serving_scope")
                    and result._scope_revision_preflight is None)
                else None),
            "applied_observable_applicability": (
                (result._slow_event or {}).get("observable_applicability")),
            "winner_program": result.winner_program,
            "winner_serving_scope": result._winner_serving_scope,
            "winner_resolved_serving_series": (
                sorted(result._winner_resolved_series)
                if result._winner_resolved_series else None),
            "winner_scope_revision": result._winner_scope_revision,
            "fast_skill_event": result._fast_skill_event,
            "llm_calls_so_far": getattr(backend, "calls", None),
        }

        if result._winner_steps:
            gate = _delayed_gate(executor, result._winner_steps, origin + 48,
                                 result._winner_serving_scope, resolve)
            entry["delayed_gate"] = gate
            loop.open_delayed(result, executor, delayed_origin=origin + 48,
                              store=store, scope_resolver=resolve)
            entry["delayed_event"] = result._delayed_event
            entry["delayed_serving_series"] = (
                sorted(result.delayed_serving_series)
                if result.delayed_serving_series else None)
            entry["delayed_scope_reresolved"] = result.delayed_scope_reresolved
            # Both must hold: the method's own delayed verdict, and the four
            # declared lines.  Activating on the first alone would let a
            # revision through on a criterion this protocol did not declare.
            if gate["passes"]:
                entry["activated"] = bool(loop.activate_approved(result, store))
            else:
                entry["activated"] = False
                entry["not_activated_because"] = [
                    name for name, ok in gate["lines"].items() if not ok]
        else:
            entry["delayed_gate"] = None
            entry["activated"] = False

        entry["skills_after"] = v1._skill_rows(method._active_snapshot())
        if entry.get("activated"):
            activated.extend(
                row["skill_id"] for row in entry["skills_after"]
                if row["skill_id"] not in activated)
        rounds.append(entry)
        print("  origin %-5d refusals=%d trigger=%-16s slow=%-24s "
              "delayed=%s activated=%s" % (
                  origin, result.risk_refusal_count, result._slow_trigger,
                  (result._slow_event or {}).get("stage"),
                  (entry.get("delayed_gate") or {}).get("passes"),
                  entry.get("activated")), flush=True)

    # Re-encounter is observed, not staged: once a Skill is active, later
    # rounds retrieve it on their own, and the artifact records whether they did.
    reencounters = [
        {"origin": entry["origin"],
         "retrieved": [sid for sid in entry.get("retrieved_skill_ids", ())
                       if sid in activated],
         "deployed_scope": entry.get("winner_serving_scope")}
        for entry in rounds if entry.get("retrieved_skill_ids")]
    survived = [row for row in reencounters if row["retrieved"]]

    report.update({
        "status": "COMPLETE",
        "rounds": rounds,
        "activated_skill_ids": activated,
        "re_encounters": reencounters,
        "llm_calls": getattr(backend, "calls", None),
        "wall_seconds": round(time.time() - started, 1),
        "verdict": (
            "SOURCE_SKILL_SURVIVED" if (activated and survived) else
            "SOURCE_SKILL_ACTIVATED_NO_REENCOUNTER" if activated else
            "A5_TREATMENT_EMPTY"),
        "stopping_rule_reading": contract.STOPPING_RULES["A5_TREATMENT_EMPTY"],
        "boundary": {**v2.BOUNDARY, "held_out_reads": 0},
    })
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = run(dry_run=args.dry_run)
    destination = OUT if not args.dry_run else OUT.with_suffix(".dry_run.json")
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
