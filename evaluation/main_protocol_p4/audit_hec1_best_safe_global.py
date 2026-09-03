"""Best-Safe-Global baseline: the best *global* program that stays inside budget.

Why it is not called an oracle
------------------------------
sol's correction, and it is not a naming quibble.  An oracle is an upper bound;
a thing a Scoped policy can beat and produce a negative regret against is not
one.  What this computes is the strongest single program applied to **every**
served series that still clears the risk budget -- identity when none does --
and arms are then reported as an *advantage* over it.  Beating it is exactly
what the serving-side line set out to show is possible, so the reading has to be
allowed to come out positive.

A real oracle would have to cover the Scoped policy -- per-UID selection, the
+0.6106 caliber in ``AGENTS`` §5.1 -- and it is reported only as an offline upper
bound, behind the oracle wall, entering no arm.

When this runs, and what it costs
---------------------------------
After the course, 0 LLM, on the evaluation face only.  Each (unit, program) pair
costs two Ridge fits, so a full menu over 26 units is thousands of fits and is
**not** run as a side effect of anything: ``--units`` defaults to zero, which
lists the plan and the fit bill and evaluates nothing.  The bill is a number the
user approves, not a surprise in a log.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/hec1_best_safe_global.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/hec1_best_safe_global.md"

#: The pair family ``AGENTS`` §5.1 found positive on both faces at origin 2856.
#: Named explicitly rather than swept: a menu that grew with the result would
#: make the baseline a search over the very readings it is the reference for.
COMPOSITION_FAMILY_FIRST = "period_median_complete"

ORACLE_BANNER = (
    "ORACLE_WALL: everything in this module is computed after the course, "
    "reads the evaluation face only, and enters no arm.  No value here may be "
    "supplied to a Fast prompt, written to an Episode bank, or used to choose "
    "a program during a run."
)


def menu() -> list[dict[str, Any]]:
    """Frozen single operators, plus the one composition family with evidence."""
    ops = list(forecast_p1._eligible_programs())
    programs: list[dict[str, Any]] = [
        {"program_id": "identity", "steps": []}]
    for op in ops:
        programs.append({
            "program_id": str(op),
            "steps": [{"op": str(op), "params": dict(forecast_p1._params(op))}],
        })
    for second in ops:
        if str(second) == COMPOSITION_FAMILY_FIRST:
            continue
        programs.append({
            "program_id": "%s>%s" % (COMPOSITION_FAMILY_FIRST, second),
            "steps": [
                {"op": COMPOSITION_FAMILY_FIRST,
                 "params": dict(forecast_p1._params(COMPOSITION_FAMILY_FIRST))},
                {"op": str(second), "params": dict(forecast_p1._params(second))},
            ],
        })
    return programs


def _steps(program: Mapping[str, Any]) -> tuple[tuple[str, dict], ...]:
    return tuple((str(step["op"]), dict(step["params"]))
                 for step in program["steps"])


def _clears(reading: Mapping[str, Any]) -> bool:
    return (float(reading["aggregate_gain"]) >= contract.RISK["material"]
            and float(reading["harmed_fraction"])
            <= contract.RISK["max_harmed_fraction"]
            and float(reading["max_single_series_harm"])
            <= contract.RISK["max_single_series_harm"])


def evaluate_unit(unit: Mapping[str, Any],
                  programs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The best in-budget global program on one unit's evaluation face."""
    ctx = runner.UnitContext(unit)
    face_origin = ctx.face_origin(runner.EVALUATION_OFFSET)
    everyone = frozenset(ctx.eval_uids)
    rows: list[dict[str, Any]] = []
    fits = 0
    for program in programs:
        steps = _steps(program)
        if not steps:
            rows.append({"program_id": program["program_id"],
                         "aggregate_gain": 0.0, "harmed_fraction": 0.0,
                         "max_single_series_harm": 0.0, "clears": True,
                         "identity": True})
            continue
        try:
            reading = runner._policy_reading(ctx, face_origin, steps, everyone)
        except runner.UnitFault as exc:
            rows.append({"program_id": program["program_id"],
                         "unusable": str(exc)[:160]})
            continue
        fits += int(reading["consumer_fits"])
        rows.append({
            "program_id": program["program_id"],
            "aggregate_gain": reading["aggregate_gain"],
            "harmed_fraction": reading["harmed_fraction"],
            "max_single_series_harm": reading["max_single_series_harm"],
            "clears": _clears(reading),
            "identity": False,
        })
    feasible = [row for row in rows if row.get("clears")
                and not row.get("identity")]
    best = (max(feasible, key=lambda row: row["aggregate_gain"])
            if feasible else {"program_id": "identity", "aggregate_gain": 0.0,
                              "identity": True})
    # The true oracle: per-series best over the same menu.  An upper bound the
    # Scoped policy can be measured against, reported and never deployed.
    return {
        "unit": ctx.unit,
        "evaluation_origin": face_origin,
        "served": len(ctx.eval_uids),
        "programs_evaluated": len(rows),
        "in_budget": len(feasible),
        "best_safe_global": best,
        "identity_when_none_clears": not feasible,
        "consumer_fits": fits,
        "readings": rows,
    }


def build(*, limit: int, ordering_name: str) -> dict[str, Any]:
    started = time.time()
    programs = menu()
    units = contract.ordering(ordering_name)
    planned = units[:max(0, int(limit))]
    rows = [evaluate_unit(unit, programs) for unit in planned]
    fits = sum(int(row["consumer_fits"]) for row in rows)
    per_unit = 2 * max(1, len([p for p in programs if p["steps"]]))
    return {
        "stage": "HEC1_BEST_SAFE_GLOBAL",
        "written_at": datetime.now().astimezone().isoformat(),
        "oracle_banner": ORACLE_BANNER,
        "contract_version": contract.VERSION,
        "data_version": contract.DATA_VERSION,
        "naming": {
            "is_not_an_oracle": (
                "a Scoped policy can beat it and produce a negative regret; "
                "the reported quantity is an advantage over a baseline"
            ),
            "true_oracle": (
                "must cover the Scoped policy (per-UID selection); offline "
                "upper bound only"
            ),
        },
        "menu": {
            "single_operators": len(
                [p for p in programs if len(p["steps"]) == 1]),
            "composition_family": COMPOSITION_FAMILY_FIRST,
            "compositions": len([p for p in programs if len(p["steps"]) == 2]),
            "total": len(programs),
            "why_this_menu": (
                "frozen P1 single operators plus the one composition family "
                "AGENTS 5.1 measured positive on both faces; a menu that grew "
                "with the result would make the reference a search"
            ),
        },
        "ordering": ordering_name,
        "units_available": len(units),
        "units_evaluated": len(rows),
        "fit_bill": {
            "per_unit_estimate": per_unit,
            "for_all_units": per_unit * len(units),
            "spent_here": fits,
            "why_default_is_zero": (
                "a full menu over every unit is thousands of Ridge fits; the "
                "bill is approved, not discovered in a log"
            ),
        },
        "units": rows,
        "boundary": {
            "llm_calls": 0,
            "baseline_fits": fits,
            "held_out_reads": 0,
            "enters_any_arm": False,
            "reads": "the evaluation face only",
        },
        "wall_seconds": round(time.time() - started, 1),
        "status": "COMPLETE" if rows else "PLAN_ONLY_NOTHING_EVALUATED",
    }


def _md(payload: Mapping[str, Any]) -> str:
    lines = [
        "# HEC-1 Best-Safe-Global baseline",
        "",
        payload["oracle_banner"],
        "",
        "Not an oracle: a Scoped policy can beat it. The reported quantity is "
        "each arm's **advantage** over it.",
        "",
        "| item | value |",
        "| --- | --- |",
        "| menu size | %s (%s single, %s compositions) |" % (
            payload["menu"]["total"], payload["menu"]["single_operators"],
            payload["menu"]["compositions"]),
        "| ordering | %s |" % payload["ordering"],
        "| units available / evaluated | %s / %s |" % (
            payload["units_available"], payload["units_evaluated"]),
        "| fits per unit (estimate) | %s |" % payload["fit_bill"][
            "per_unit_estimate"],
        "| fits for all units | %s |" % payload["fit_bill"]["for_all_units"],
        "| fits spent here | %s |" % payload["fit_bill"]["spent_here"],
        "| status | %s |" % payload["status"],
        "",
    ]
    if payload["units"]:
        lines += ["| unit | best in-budget program | aggregate | identity |",
                  "| --- | --- | ---: | --- |"]
        for row in payload["units"]:
            best = row["best_safe_global"]
            lines.append("| %s x %s | `%s` | %+.4f | %s |" % (
                row["unit"]["block"], row["unit"]["origin"],
                best["program_id"], float(best["aggregate_gain"]),
                row["identity_when_none_clears"]))
        lines.append("")
    else:
        lines += [
            "Nothing was evaluated: `--units` defaults to 0 so the fit bill is "
            "declared before it is spent.", ""]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=0,
                       help="how many units to evaluate; 0 lists the plan only")
    parser.add_argument("--ordering", choices=list(contract.ORDERINGS),
                       default="forward")
    args = parser.parse_args(argv)
    payload = build(limit=args.units, ordering_name=args.ordering)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    print("menu programs   : %s" % payload["menu"]["total"])
    print("units evaluated : %s / %s"
          % (payload["units_evaluated"], payload["units_available"]))
    print("baseline fits   : %s (all units would be ~%s)"
          % (payload["boundary"]["baseline_fits"],
             payload["fit_bill"]["for_all_units"]))
    print("status          : %s" % payload["status"])
    print("wrote %s" % OUT_JSON.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
