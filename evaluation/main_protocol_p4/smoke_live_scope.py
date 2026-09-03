"""One cell, one real Fast call: does the Scope chain close end to end?

Everything before this ran on a sealed backend.  What could not be answered
offline is whether the chain still holds when the Program is proposed by a real
Agent rather than by a stub:

    real Fast proposes a Program
    -> the frozen Runtime initialiser turns it into a Scope
    -> the Scope resolves to a legal proper subset of the served series
    -> that subset actually enters the serving-side evaluator

The Fast Agent is *not* asked to emit a Scope: its schema has no channel for one
and this round deliberately does not add a field no instruction mentions.  The
claim on trial is that the Harness forms a Scope, not that a single Fast call
does.

It is a smoke test.  A Scope that performs badly still passes -- utility is what
Static / A3 / A5 is for, and none of those may be changed by what is seen here.
Held-out stays closed; one held-in cell, a handful of LLM calls.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import audit_cross_fitted_targeting as targeting
from evaluation.main_protocol_p4 import audit_program_repairability as p4c
from evaluation.main_protocol_p4 import main_experiment_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import scope_initializer as initializer
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4r_live_scope_smoke.json"

REQUIRED_ENV = ("M0_AGENT_BASE_URL", "M0_AGENT_MODEL")
KEY_ENV = ("CPA_API_KEY", "AGICTO_API_KEY", "OPENAI_API_KEY")
SMOKE_ORIGIN = contract.HELD_IN_ORIGINS[0]
SMOKE_FACE = "support_a"
#: The frozen per-arm LLM budget; the Fast flow is inspect -> propose ->
#: select, so a smaller cap exhausts mid-flow and looks like an abstention.
MAX_CALLS = contract.PER_ARM_BUDGET["llm_calls"]


def transport() -> dict[str, Any]:
    """What the run would use, and whether it is complete enough to run."""
    settings = {name: os.environ.get(name) for name in REQUIRED_ENV}
    key_name = next((name for name in KEY_ENV if os.environ.get(name)), None)
    missing = [name for name, value in settings.items() if not value]
    return {
        "base_url": settings["M0_AGENT_BASE_URL"],
        "model": settings["M0_AGENT_MODEL"],
        "key_source": key_name,
        "missing": missing + ([] if key_name else ["one of %s" % (KEY_ENV,)]),
        "ready": not missing and key_name is not None,
        "why_no_default": (
            "an unset base URL would send a key to whatever endpoint the client "
            "defaults to; a missing setting is a refusal, not a fallback"
        ),
    }


def _feature_cards(variant: Any, uids: Sequence[str],
                   origin: int) -> dict[str, dict[str, float]]:
    """Per-series deployment-visible features, taken strictly pre-origin."""
    matrix, names = targeting.series_features(variant, list(uids), int(origin))
    return {
        uid: {name: float(value) for name, value in zip(names, row)}
        for uid, row in zip(uids, matrix)
    }


def _fast_candidates(series: np.ndarray, values: Any, origin: int,
                     domain: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One real Fast proposal.  The Agent returns Programs; Scopes are not asked for."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
    sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))
    import run_v1_guidance_evolution as runner  # noqa: PLC0415

    from evaluation.functional.task_episode_harness.agentic import (  # noqa: PLC0415
        runner as agentic,
    )
    from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: PLC0415
        TTHAAgentCore,
    )
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: PLC0415
        TTHAFastAgent,
    )
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: PLC0415
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: PLC0415
        np_values,
    )
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: PLC0415
        LocalPublicToolGateway,
    )

    target = agentic.live_transport()
    backend = agentic._default_backend_factory(MAX_CALLS)
    # The core stamps its own base_url and model onto every request and the
    # backend refuses a mismatch.  Leaving them at the module defaults sends an
    # agicto-addressed request to a nowaterapi backend, which fails as
    # "compilation_status: failed" with no candidates and no tool calls -- a
    # harness misconfiguration that looks exactly like an Agent that abstained.
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series[:origin], task_kind="forecast"),
        model=target["model"], base_url=target["base_url"])
    method = TTHAMethod(TTHAFastAgent(core), runner._h0_snapshot(), ())
    request = runner._a5_request(series, values, origin, domain)
    method.bind_round_data(
        np_values(request, values)[:origin],
        task_kind=request.task_spec.task_type)
    method.prepare(request)
    trace = method.last_trace
    steps_map = dict(getattr(trace, "candidate_program_steps", {}) or {})
    return (
        [{"candidate_id": cand, "steps": [
            {"op": op, "params": dict(params)} for op, params in steps]}
         for cand, steps in steps_map.items()],
        {"transport": target, "calls_made": getattr(backend, "calls", None)},
    )


def build(*, dry_run: bool) -> dict[str, Any]:
    started = time.time()
    state = transport()
    report: dict[str, Any] = {
        "stage": "P4R_LIVE_SCOPE_SMOKE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_AGENT_CAPABILITY_SMOKE",
        "data_version": contract.DATA_VERSION,
        "question": (
            "does the chain close with a real Fast call: Program proposed, "
            "Scope initialised by the frozen Runtime rule, resolved to a legal "
            "proper subset, and actually executed by the serving evaluator"
        ),
        "not_on_trial": [
            "utility: a legal Scope that performs badly still passes",
            "Fast emitting a Scope: its schema has no channel and none was added",
        ],
        "scope_origination": contract.SCOPE_ORIGINATION,
        "initializer_rules": initializer.declared_rules(),
        "cell": {"origin": int(SMOKE_ORIGIN), "face": SMOKE_FACE},
        "transport": state,
    }
    if dry_run or not state["ready"]:
        report["status"] = "DRY_RUN" if state["ready"] else "BLOCKED_ON_TRANSPORT"
        report["llm_calls"] = 0
        report["verdict"] = "DRY_RUN_OK" if state["ready"] else "TRANSPORT_NOT_CONFIGURED"
        return report

    groups = contract.cohorts()
    cell, variant = baselines._cell(groups["target"])
    at = forecast_p4._cell_at(cell, int(SMOKE_ORIGIN))
    config = forecast_p4._config(int(SMOKE_ORIGIN))
    roster = at.roster(SMOKE_FACE)
    eval_uids = [str(row["series_uid"]) for row in roster if row["role"] == "eval"]
    cards = _feature_cards(variant, eval_uids, int(SMOKE_ORIGIN))

    candidates, meta = _fast_candidates(
        at.observation_block, at.values, int(SMOKE_ORIGIN), "p4r-scope-smoke")
    report["fast_proposal"] = {
        "candidates": [c["candidate_id"] for c in candidates],
        "programs": {c["candidate_id"]: [s["op"] for s in c["steps"]]
                     for c in candidates},
        **meta,
    }

    executor = p4c.ScopeExecutor(
        roster, at.values, config, evaluate_fn=views.forecast_runtime._evaluate,
        max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION)
    rows, fits = [], 0
    for candidate in candidates:
        init = initializer.initialize(candidate["steps"])
        resolved = initializer.resolve(init["scope"], cards)
        row: dict[str, Any] = {
            "candidate_id": candidate["candidate_id"],
            "program": [s["op"] for s in candidate["steps"]],
            **init,
            "resolved_count": len(resolved),
            "served_count": len(eval_uids),
            "proper_subset": 0 < len(resolved) < len(eval_uids),
        }
        steps = tuple((str(s["op"]), dict(s["params"])) for s in candidate["steps"])
        if not resolved:
            row["executed"] = False
            row["why"] = "the Scope selected nobody; nothing to execute"
        elif not executor.verify(steps, int(SMOKE_ORIGIN)).passed:
            row["executed"] = False
            row["why"] = "WINDOW_VERIFIER_REJECTED"
        else:
            try:
                reading = scoped.scoped_evaluate(
                    roster, at.values, executor._compiled(steps), config,
                    origin=int(SMOKE_ORIGIN), scope=resolved)
                fits += reading["consumer_fits"]
                gains = np.asarray(reading["per_view_smase"], dtype=np.float64)
                static = scoped.scoped_evaluate(
                    roster, at.values, None, config, origin=int(SMOKE_ORIGIN))
                fits += static["consumer_fits"]
                base = np.asarray(static["per_view_smase"], dtype=np.float64)
                delta = base - gains
                untouched = np.array([uid not in resolved for uid in eval_uids])
                row.update({
                    "executed": True,
                    "mean_gain": round(float(delta.mean()), 6),
                    "declined_series_all_exactly_static": bool(
                        np.array_equal(delta[untouched], np.zeros(int(untouched.sum())))),
                    "treated_series_that_moved": int(np.count_nonzero(
                        np.abs(delta[~untouched]) > 1e-12)),
                })
            except scoped.ServingContextDegenerate as exc:
                row["executed"] = False
                row["why"] = "SERVING_CONTEXT_DEGENERATE: %s" % str(exc)[:120]
            except Exception as exc:  # noqa: BLE001
                row["executed"] = False
                row["why"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
        rows.append(row)

    executed = [row for row in rows if row.get("executed")]
    legal = [row for row in executed if row["proper_subset"]]
    report.update({
        "status": "COMPLETE",
        "candidates": rows,
        "boundary": {
            "cells": 1, "consumer_fits": fits, "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
        },
        "wall_seconds": round(time.time() - started, 1),
        "verdict": (
            "SCOPE_CHAIN_CLOSES_WITH_A_REAL_FAST_CALL" if legal
            else "SCOPE_CHAIN_DID_NOT_CLOSE"
        ),
        "releases": (
            "the Source line may start" if legal
            else "nothing; the chain must close before A3/A5"
        ),
    })
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = build(dry_run=args.dry_run)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("transport ready : %s" % report["transport"]["ready"])
    for row in report.get("candidates", []):
        print("  %-22s %-34s -> %2d/%2d  subset=%-5s executed=%-5s %s" % (
            row["candidate_id"], ">".join(row["program"])[:34],
            row["resolved_count"], row["served_count"],
            row["proper_subset"], row.get("executed"),
            row.get("why", "gain %+.4f" % row["mean_gain"]
                    if row.get("executed") else "")))
    print("verdict         : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
