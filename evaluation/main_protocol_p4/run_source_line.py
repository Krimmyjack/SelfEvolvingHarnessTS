"""Step 5 of the frozen run order: form and approve Skills on the Source cohort.

A5 is "audited Source Skills, then Target held-in calibration".  Its treatment
therefore has to exist before it runs, and it has to be formed the way the
project claims Skills are formed -- by the Harness, on a cohort of its own, from
its own feedback -- not written by hand into a fixture.  That is all this runner
does.  It is **not an arm**: nothing it measures enters the A3/A5 comparison,
and its cost is billed separately from ``PER_ARM_BUDGET``.

The lifecycle it drives is the one already wired:

    real Fast proposes Programs
    -> the frozen Runtime initialiser gives each candidate a Scope
    -> the Scope resolves against deployment-visible features at this origin
    -> the scoped serving evaluator probes it, and the winner becomes a Draft
       Skill carrying the **predicate** (never the resolved UIDs)
    -> the delayed gate at origin+48 approves or refuses it

Two things about that gate are worth stating, because both were wrong or
ambiguous until this step:

* the delayed reading is now taken **under the winner's own Scope**, re-resolved
  at the delayed origin.  Judging a scoped Skill on a global average asks it to
  answer for series it never proposed to touch -- the same "one program applied
  globally" reading this whole line was moved away from;
* ``online_loop.current_status`` calls a capability Skill *active* only when its
  ``observable_applicability`` is narrower than ``task_kind`` alone, and *draft*
  otherwise.  The natural forecast line has always emitted the wide signature
  (``run_e2_t6_natural_a5_a3`` does), so an approved Source Skill lands in
  ``draft_skills``.  Draft is not empty: a Draft Skill is retrieved and supplied
  at the Target, it merely does not keep a priority slot until Target Support
  confirms it.  Both counts are reported and the stopping rule is read on
  *approved* Skills, with that reading disclosed rather than assumed.

Source is ``readable[160:200]``; the origins are the held-in block and the
delayed origin is ``origin + 48``, so the furthest read is 2904 -- the held-out
block at 4056+ is not touched.
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
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import scope_initializer as initializer
from evaluation.main_protocol_p4 import smoke_live_scope as smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4w_source_line.json"

FACE = "support_a"
HORIZON = contract.HORIZON
DOMAIN = "p4w-source-line"

#: The admission rule the arms run under.  ``admission_policy.DEFAULT`` is
#: ``strict_positive_only``: a probe earns deployment rights only if *no* series
#: is materially harmed.  The adjudicated operating point for this protocol is
#: BOUNDED_V1 at 0.20 / 0.30, and the main contract's own endpoints are stated
#: in those two numbers -- so leaving the library default in force would gate
#: every arm on one rule while reporting it against another.  Installing the
#: already-frozen ``p4b_contract.BOUNDED_POLICY`` changes no threshold; it
#: stops the runner from silently using a rule the protocol never chose.
ADMISSION_POLICY = {
    "rule": bounded.BOUNDED_POLICY.rule,
    "max_harmed_fraction": bounded.BOUNDED_MAX_HARMED_FRACTION,
    "max_single_series_harm": bounded.BOUNDED_MAX_SINGLE_SERIES_HARM,
    "library_default_would_have_been": "strict_positive_only",
    "thresholds_changed": 0,
}

#: The Source line is not an arm, so it draws on its own declared budget rather
#: than widening the frozen per-arm vector.  Five rounds of inspect / propose /
#: select plus headroom for a retry; no Slow agent is called.
SOURCE_BUDGET = {
    "llm_calls": 40,
    "rounds": len(contract.HELD_IN_ORIGINS),
    "face": FACE,
    "is_an_arm": False,
    "enters_a3_a5_comparison": False,
}


class InitializerScopes(Mapping):
    """The frozen initialiser, evaluated on demand instead of tabled ahead.

    ``run_online_round`` takes candidate Scopes as a mapping because a Runner
    that knows its candidates can write the table before the round.  A real Fast
    call names its own candidates, so no such table can exist; this view answers
    each lookup by running the same frozen rule on that candidate's steps.  The
    Runner still cannot hand-pick: it has no branch on the candidate id, only on
    what the Agent proposed.
    """

    def __init__(self, method: Any) -> None:
        self._method = method

    def _steps(self) -> dict[str, Any]:
        trace = getattr(self._method, "last_trace", None)
        return dict(getattr(trace, "candidate_program_steps", {}) or {})

    def __getitem__(self, candidate_id: str) -> Mapping[str, Any]:
        return initializer.initialize(self._steps()[candidate_id])["scope"]

    def __iter__(self):
        return iter(self._steps())

    def __len__(self) -> int:
        return len(self._steps())

    def __bool__(self) -> bool:
        # An empty trace must not make the whole Scope channel fall away:
        # ``candidate_scopes or {}`` upstream would silently drop it.
        return True


def _resolver(variant: Mapping[str, Any], eval_uids: Sequence[str]):
    """Resolve a predicate against this cell's own pre-origin features.

    Cached per origin, never across origins: the predicate is re-read at every
    decision point precisely because the structure it names can change.
    """
    cache: dict[int, dict[str, dict[str, float]]] = {}

    def resolve(spec: Mapping[str, Any], origin: int) -> frozenset[str]:
        key = int(origin)
        if key not in cache:
            cache[key] = smoke._feature_cards(variant, list(eval_uids), key)
        return initializer.resolve(spec, cache[key])

    return resolve


def _machinery() -> dict[str, Any]:
    """Import the method chain once, with the evaluation paths it expects."""
    sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
    sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))
    import run_v1_guidance_evolution as runner  # noqa: PLC0415

    from evaluation.functional.task_episode_harness.agentic import (  # noqa: PLC0415
        runner as agentic,
    )
    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: PLC0415
        EditController,
        FaultRouter,
        SurfaceRegistry,
    )
    from SelfEvolvingHarnessTS.methods.ttha import (  # noqa: PLC0415
        admission_policy,
        online_loop,
    )
    from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: PLC0415
        TTHAAgentCore,
    )
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: PLC0415
        TTHAFastAgent,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: PLC0415
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: PLC0415
        SnapshotStore,
    )
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: PLC0415
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: PLC0415
        LocalPublicToolGateway,
        extract_public_features,
    )
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: PLC0415
        ScopeExecutor,
    )

    return {
        "runner": runner, "agentic": agentic,
        "EditController": EditController, "FaultRouter": FaultRouter,
        "SurfaceRegistry": SurfaceRegistry, "TTHAAgentCore": TTHAAgentCore,
        "TTHAFastAgent": TTHAFastAgent, "compile_snapshot": compile_snapshot,
        "SnapshotStore": SnapshotStore, "TTHAMethod": TTHAMethod,
        "online_loop": online_loop, "admission_policy": admission_policy,
        "LocalPublicToolGateway": LocalPublicToolGateway,
        "extract_public_features": extract_public_features,
        "ScopeExecutor": ScopeExecutor,
    }


def _skill_rows(snapshot: Any) -> list[dict[str, Any]]:
    """What the snapshot now carries, including each Skill's own predicate."""
    rows = []
    for skill in list(getattr(snapshot, "skills", ()) or ()):
        kind = str(getattr(getattr(skill, "skill_kind", None), "value", ""))
        if kind != "capability":
            continue
        guards = dict(skill.risk_guards or {})
        scope = getattr(skill, "serving_scope", None)
        rows.append({
            "skill_id": skill.skill_id,
            "draft": guards.get("requires_target_support") is True,
            "serving_scope": dict(scope) if scope else None,
            "allowed_tools": list(getattr(skill, "allowed_tools", ()) or ()),
            "observable_applicability": dict(
                getattr(skill, "observable_applicability", {}) or {}),
        })
    return rows


def _card(_episode: Any) -> dict[str, Any]:
    """The signature the natural forecast line has always emitted.

    Widening or narrowing it here would be the Runner choosing how easily its
    own Skills are retrieved, which is exactly what the frozen rules forbid.
    """
    return {
        "pattern_id": "p4w-source-line",
        "observable_signature": {"task_kind": "forecast"},
    }


def run(*, dry_run: bool) -> dict[str, Any]:
    started = time.time()
    state = contract.assert_frozen()
    if not state["frozen"]:
        raise RuntimeError(
            "main experiment contract drifted: %s" % state["failures"])
    transport = smoke.transport()
    report: dict[str, Any] = {
        "stage": "P4W_SOURCE_LINE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_SOURCE_SKILL_FORMATION",
        "data_version": contract.DATA_VERSION,
        "question": (
            "does the Harness form and approve Skills on the Source cohort, so "
            "that A5 has a treatment that was not written by hand"
        ),
        "not_on_trial": [
            "A5 > A3: nothing measured here enters that comparison",
            "utility on the Target: this cohort is never scored there",
        ],
        "cohort": "source readable[%d:%d]" % contract.SOURCE_SLICE,
        "origins": list(contract.HELD_IN_ORIGINS),
        "delayed_origin_rule": "origin + %d" % HORIZON,
        "source_budget": SOURCE_BUDGET,
        "admission_policy": ADMISSION_POLICY,
        "scope_origination": contract.SCOPE_ORIGINATION,
        "initializer_rules": initializer.declared_rules(),
        "transport": transport,
    }
    if dry_run or not transport["ready"]:
        report["status"] = ("DRY_RUN" if transport["ready"]
                            else "BLOCKED_ON_TRANSPORT")
        report["llm_calls"] = 0
        report["verdict"] = ("DRY_RUN_OK" if transport["ready"]
                             else "TRANSPORT_NOT_CONFIGURED")
        return report

    m = _machinery()
    m["admission_policy"].install_policy(bounded.BOUNDED_POLICY)
    groups = contract.cohorts()
    cell, variant = baselines._cell(groups["source"])
    series0 = np.asarray(cell.values[cell.support_a[0]], dtype=np.float64)

    store = m["SnapshotStore"](PROJECT_ROOT / ".p4w_source_store")
    controller = m["EditController"](
        store, surfaces=m["SurfaceRegistry"](), router=m["FaultRouter"]())
    h0 = m["compile_snapshot"](
        PROJECT_ROOT / "methods/ttha/harness/h0", verify_lock=False)

    backend = m["agentic"]._default_backend_factory(SOURCE_BUDGET["llm_calls"])
    target = m["agentic"].live_transport()
    core = m["TTHAAgentCore"](
        backend,
        m["LocalPublicToolGateway"](
            series0[:contract.HELD_IN_ORIGINS[0]], task_kind="forecast"),
        model=target["model"], base_url=target["base_url"])
    method = m["TTHAMethod"](m["TTHAFastAgent"](core), h0, ())

    rounds: list[dict[str, Any]] = []
    approved: list[str] = []
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
        resolve = _resolver(variant, eval_uids)
        core.tools = m["LocalPublicToolGateway"](
            series0[:origin], task_kind="forecast")
        features = dict(m["extract_public_features"](
            series0[:origin], task_kind="forecast"))
        try:
            result = m["online_loop"].run_online_round(
                method, executor,
                m["runner"]._a5_request(series0, at.values, origin, DOMAIN),
                at.values,
                origin=origin,
                slow_agent=None, controller=controller, store=store,
                card_builder=_card,
                round_name="p4w_source_r%d" % (index + 1),
                budget=contract.PER_ARM_BUDGET["probes"],
                allow_slow=False, allow_group_slow=False,
                domain=DOMAIN, period=int(config["period"]),
                fast_features=features,
                allow_fast_skill=True,
                candidate_scopes=InitializerScopes(method),
                scope_resolver=resolve,
            )
        except Exception as exc:  # noqa: BLE001 - a blocked round is a reading
            rounds.append({
                "origin": origin,
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:240]),
                "llm_calls_so_far": getattr(backend, "calls", None),
            })
            print("  origin %d BLOCKED %s" % (origin, type(exc).__name__),
                  flush=True)
            break
        m["online_loop"].open_delayed(
            result, executor, delayed_origin=origin + HORIZON,
            store=store, scope_resolver=resolve)
        if result.approved_skill_id is not None:
            m["online_loop"].activate_approved(result, store)
            approved.append(str(result.approved_skill_id))
        trace = method.last_trace
        rounds.append({
            "origin": origin,
            "candidate_ids": list(trace.candidate_ids or ()),
            "chosen_candidate_id": str(trace.chosen_candidate_id or ""),
            "retrieved_skill_ids": list(trace.retrieved_skill_ids or ()),
            "winner_program": result.winner_program,
            "winner_serving_scope": result._winner_serving_scope,
            "winner_resolved_count": (
                None if result._winner_resolved_series is None
                else len(result._winner_resolved_series)),
            "served_count": len(eval_uids),
            # The whole row: gain alone cannot tell a refused deployment from
            # an Agent that proposed nothing, and that is exactly the confusion
            # the first pass of this runner produced.
            "probes": [dict(probe) for probe in result.actual_probed_programs],
            "first_positive_support_receipt_index": (
                result.first_positive_support_receipt_index),
            "harm_count": result.harm_count,
            # 聚合过线、被尾部预算拒绝的候选。此前这类事件不进任何计数器，
            # 一轮全是风险拒绝时账面上是零故障——正是 p4w 第一遍读错的原因。
            "risk_refusal_count": result.risk_refusal_count,
            "risk_refusals": [dict(row) for row in result.risk_refusals],
            "slow_trigger": result._slow_trigger,
            "slow_event": result._slow_event,
            "fast_skill_event": result._fast_skill_event,
            "delayed_event": result._delayed_event,
            "delayed_utility": result.delayed_utility,
            "delayed_scope_reresolved": result.delayed_scope_reresolved,
            "delayed_serving_series_count": (
                None if result.delayed_serving_series is None
                else len(result.delayed_serving_series)),
            "approved_skill_id": result.approved_skill_id,
            "snapshot_sha": method._active_snapshot().harness_content_sha,
            "llm_calls_so_far": getattr(backend, "calls", None),
        })
        print("  origin %d: winner=%s approved=%s calls=%s" % (
            origin, result.winner_program, result.approved_skill_id,
            getattr(backend, "calls", None)), flush=True)

    status = m["online_loop"].current_status(store, method)
    skills = _skill_rows(method._active_snapshot())
    formed = [row for row in skills if row["skill_id"] in set(approved)]
    report.update({
        "status": "COMPLETE",
        "rounds": rounds,
        "approved_skill_ids": approved,
        "skills_in_snapshot": skills,
        "source_skills_for_a5": formed,
        "runtime_status": {
            "active_skills": status["active_skills"],
            "draft_skills": status["draft_skills"],
            "episodes_count": status["episodes_count"],
        },
        "stopping_rule_reading": {
            "rule": contract.STOPPING_RULES["A5_TREATMENT_EMPTY"],
            "evaluated_on": "approved Source Skills",
            "approved_count": len(approved),
            "why_not_current_status_active_skills": (
                "current_status calls a capability Skill active only when its "
                "observable_applicability is narrower than task_kind alone; the "
                "natural forecast line has always emitted the wide signature, so "
                "an approved Skill lands in draft_skills.  A Draft Skill is still "
                "retrieved and supplied at the Target -- it only loses its "
                "priority slot until Target Support confirms it -- so reading "
                "draft as no treatment would stop A5 for a naming reason"
            ),
        },
        "boundary": {
            **contract.BOUNDARY,
            "llm_calls": getattr(backend, "calls", None),
            "held_out_reads": 0,
            "furthest_origin_read": int(contract.HELD_IN_ORIGINS[-1]) + HORIZON,
            "held_out_block_starts_at": int(contract.HELD_OUT_ORIGINS[0]),
        },
        "wall_seconds": round(time.time() - started, 1),
        "verdict": ("SOURCE_SKILLS_FORMED" if approved else "A5_TREATMENT_EMPTY"),
        "admission_policy_in_force": m["admission_policy"].active_policy().rule,
        "releases": ("A5 may carry these Skills to the Target" if approved
                     else "nothing; A5 must not be run as if it had a treatment"),
    })
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = run(dry_run=args.dry_run)
    # 一次 --dry-run 曾把一次真跑的工件整个盖掉（逐序列 gain 与解析 UID
    # 无法从任何地方恢复）。写路径因此按运行种类分开，且真跑不覆盖已存在
    # 的真工件——要重跑先显式让路。"不覆盖历史结果"是规则，不能靠记得。
    destination = OUT if not args.dry_run else OUT.with_suffix(".dry_run.json")
    if destination.exists() and not args.dry_run:
        raise SystemExit(
            "refusing to overwrite an existing live artifact: %s\n"
            "move or rename it first; a run that cannot be reproduced must not "
            "be destroyed by the next one"
            % destination.relative_to(PROJECT_ROOT).as_posix())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print("transport ready : %s" % report["transport"]["ready"])
    for row in report.get("rounds", ()):
        if "error" in row:
            print("  origin %-5s BLOCKED %s" % (row["origin"], row["error"]))
            continue
        print("  origin %-5d winner=%-26s scope=%s/%s delayed=%-9s approved=%s"
              % (row["origin"],
                 ">".join(row["winner_program"] or []) or "identity",
                 row["winner_resolved_count"], row["served_count"],
                 (row["delayed_event"] or {}).get("stage"),
                 row["approved_skill_id"]))
    print("verdict         : %s" % report["verdict"])
    print("wrote %s" % destination.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
