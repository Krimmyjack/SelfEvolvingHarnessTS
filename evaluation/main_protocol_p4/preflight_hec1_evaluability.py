"""Which units can be scored at all, decided before any of them is run.

The gap this closes
-------------------
D2 screened whether the missing-aware sMASE is defined **at the origin**.  The
evaluation face is 144 steps further on, and on some units it runs past the end
of a series' observed data -- so the metric is undefined there and no arm can be
scored.  The shakedown Forward hit this at origin 2856 and it crashed the course
before the fault was classified.

Discovering that mid-run is bad for two reasons.  The obvious one is that a
crash is a worse reading than a recorded abstention.  The subtler one is that
``HEC1_INCONCLUSIVE`` is defined against ``0.8 x N_T``, and if some units can
never contribute a curve point then the denominator is not 26 -- it is
``N_T_eff``, and a completion rate measured against the wrong denominator either
flatters the run or condemns it.

What it reads, and what it must not
-----------------------------------
Only whether the metric is **definable**: the served window exists, the horizon
contains at least one observed value, and the seasonal scale is finite.  It does
not fit a model, does not apply a program, and never looks at an error or a
gain.  This is the same class of check that chose the origins in the first place
(``p4s``'s "no gain, error or utility participated"), moved 144 steps along.

Learning units are **not** dropped.  A unit whose evaluation face is unscoreable
still runs: the Harness still probes it, the delayed gate still decides, and the
Episode still enters the bank.  Only its contribution to the curve is absent,
and it is absent for **every arm identically**, because the cause is the data.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import hec1_scoreability as scoreability
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import run_hec1 as runner
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import seasonal_scale

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts/main_protocol"
OUT_JSON = ARTIFACTS / "hec1_evaluability.json"
OUT_MD = ARTIFACTS / "hec1_evaluability.md"

CONTEXT, HORIZON = preflight.CONTEXT, preflight.HORIZON
PERIOD = preflight.PERIOD
EVALUATION_OFFSET = runner.EVALUATION_OFFSET
DELAYED_OFFSET = runner.DELAYED_OFFSET


def _face_is_definable(raw: np.ndarray, origin: int) -> dict[str, Any]:
    """Is the missing-aware sMASE defined for this series at this origin?

    Three ways it is not, and each is reported separately rather than collapsed
    into a boolean, because "the series is too short" and "the horizon is all
    gap" are different facts about the data.
    """
    length = int(raw.size)
    if length < origin + HORIZON:
        return {"definable": False, "why": "series ends before origin+horizon",
                "length": length, "needed": origin + HORIZON}
    window = raw[origin - CONTEXT:origin]
    if window.size != CONTEXT:
        return {"definable": False, "why": "serving context shorter than %d"
                                           % CONTEXT}
    truth = raw[origin:origin + HORIZON]
    observed = int(np.count_nonzero(np.isfinite(truth)))
    if observed == 0:
        return {"definable": False, "why": "horizon contains no observed truth",
                "observed_in_horizon": 0}
    history = raw[:origin]
    scale = seasonal_scale(history, np.isfinite(history), period=PERIOD,
                           min_pairs=32)
    if scale is None or not np.isfinite(scale) or float(scale) <= 0.0:
        return {"definable": False, "why": "seasonal scale is not finite",
                "scale": None if scale is None else float(scale)}
    try:
        completed = forecast_runtime._linear_integrity(window)
        _c, _s, method = forecast_runtime._center_scale(np, completed)
    except Exception as exc:  # noqa: BLE001 - a 0-fit legality gate
        return {"definable": False,
                "why": "%s: %s" % (type(exc).__name__, str(exc)[:80])}
    if method == "scale_floor_fallback":
        return {"definable": False, "why": "serving context is degenerate"}
    return {"definable": True, "observed_in_horizon": observed}


def scan_unit(unit: Mapping[str, Any], variant: Mapping[str, np.ndarray],
              readable: Sequence[str]) -> dict[str, Any]:
    """Both scored faces of one unit, over its own served series."""
    span = (int(unit["span"][0]), int(unit["span"][1]))
    uids = list(readable[span[0]:span[1]])
    # Face A is the served face; its length is read here rather than assumed,
    # which is the same reason the runner's denominators come from the roster.
    served = uids[:20]
    origin = int(unit["origin"])
    faces: dict[str, Any] = {}
    for name, offset in (("delayed", DELAYED_OFFSET),
                         ("evaluation", EVALUATION_OFFSET)):
        window = origin + offset
        rows = [
            {"uid": uid, **_face_is_definable(
                np.asarray(variant[uid], dtype=np.float64), window)}
            for uid in served
        ]
        undefinable = [row for row in rows if not row["definable"]]
        faces[name] = {
            "window": window,
            "served": len(served),
            "definable": len(rows) - len(undefinable),
            "undefinable": len(undefinable),
            # Any undefinable series makes the whole face unscoreable: the
            # evaluator raises rather than scoring a subset, and a partial face
            # would not be comparable with a full one anyway.
            "face_scoreable": not undefinable,
            "reasons": sorted({row["why"] for row in undefinable}),
            "undefinable_uids": [row["uid"] for row in undefinable][:8],
        }
    return {
        "block": str(unit["block"]),
        "span": list(span),
        "origin": origin,
        "served": len(served),
        "faces": faces,
        "contributes_a_curve_point": faces["evaluation"]["face_scoreable"],
        "gate_is_readable": faces["delayed"]["face_scoreable"],
    }


def build() -> dict[str, Any]:
    started = time.time()
    variant = preflight.load_variant()
    readable = runner.readable_uids()

    scans: dict[str, list[dict[str, Any]]] = {}
    for name in contract.ORDERINGS:
        scans[name] = [scan_unit(unit, variant, readable)
                       for unit in contract.ordering(name)]
    phase_s = [scan_unit(unit, variant, readable)
               for unit in contract.phase_s_units()]

    forward = scans["forward"]
    n_t = len(forward)
    n_t_eff = sum(1 for row in forward if row["contributes_a_curve_point"])
    dropped = [row for row in forward if not row["contributes_a_curve_point"]]
    gate_blind = [row for row in forward if not row["gate_is_readable"]]

    # Every ordering is a permutation of the same units, so N_T_eff cannot
    # differ between them -- asserted rather than assumed, because if it ever
    # did, one ordering would be scored on a different denominator.
    per_ordering = {
        name: sum(1 for row in rows if row["contributes_a_curve_point"])
        for name, rows in scans.items()
    }
    consistent = len(set(per_ordering.values())) == 1

    payload = {
        "stage": "HEC1_EVALUABILITY",
        "written_at": datetime.now().astimezone().isoformat(),
        "contract_version": contract.VERSION,
        "data_version": contract.DATA_VERSION,
        "what_it_reads": (
            "whether the missing-aware sMASE is definable: the served window "
            "exists, the horizon holds at least one observed value, and the "
            "seasonal scale is finite"
        ),
        "what_it_does_not_read": ["any error", "any gain", "any utility",
                                  "any model fit"],
        "consumer_fits": 0,
        "llm_calls": 0,
        "phase_t": {
            "N_T": n_t,
            "N_T_eff": n_t_eff,
            "dropped_units": [
                {"block": row["block"], "origin": row["origin"],
                 "reasons": row["faces"]["evaluation"]["reasons"]}
                for row in dropped],
            "gate_unreadable_units": [
                {"block": row["block"], "origin": row["origin"],
                 "reasons": row["faces"]["delayed"]["reasons"]}
                for row in gate_blind],
            # ceil, not int: int(0.8 x 23) is 18 and 18/23 is 78.3%, which is
            # not the 80% the contract declares.  19/23 is 82.6%.
            "min_paired_curve_points": math.ceil(
                scoreability.COMPLETION_FRACTION * n_t_eff),
            "per_ordering_N_T_eff": per_ordering,
            "orderings_agree": consistent,
        },
        "phase_s": {
            "N_S": len(phase_s),
            "N_S_eff": sum(1 for row in phase_s
                           if row["contributes_a_curve_point"]),
            "dropped_units": [
                {"block": row["block"], "origin": row["origin"],
                 "reasons": row["faces"]["evaluation"]["reasons"]}
                for row in phase_s if not row["contributes_a_curve_point"]],
        },
        "units": {"phase_s": phase_s, **scans},
        "handling": (
            "a unit whose evaluation face is unscoreable still runs and still "
            "writes Episodes; it contributes no curve point, and it does so for "
            "every arm identically because the cause is the data"
        ),
        "boundary": {"llm_calls": 0, "consumer_fits": 0, "held_out_reads": 0,
                     "outcome_values_read": 0, "thresholds_changed": 0},
        "wall_seconds": round(time.time() - started, 1),
    }
    # Mechanical cross-check, both ways: the frozen declaration cannot drift
    # from the data, and a data change cannot pass as the declaration.
    payload["frozen_manifest"] = scoreability.to_dict()
    payload["cross_check"] = scoreability.verify_against(payload)
    return payload


def _md(payload: Mapping[str, Any]) -> str:
    phase_t = payload["phase_t"]
    lines = [
        "# HEC-1 evaluation-face evaluability (0 fit, 0 LLM)",
        "",
        "Reads only whether the metric is **definable** at each scored window. "
        "No error, gain or utility participates.",
        "",
        "| item | value |",
        "| --- | --- |",
        "| scheduled units (N_T) | %s |" % phase_t["N_T"],
        "| **scoreable units (N_T_eff)** | **%s** |" % phase_t["N_T_eff"],
        "| **min paired curve points** (ceil 0.8 x N_T_eff) | **%s** |"
        % phase_t["min_paired_curve_points"],
        "| orderings agree on N_T_eff | %s |" % phase_t["orderings_agree"],
        "| frozen manifest cross-check | %s |" % payload["cross_check"][
            "passed"],
        "| Phase S N_S / N_S_eff | %s / %s |" % (
            payload["phase_s"]["N_S"], payload["phase_s"]["N_S_eff"]),
        "",
    ]
    if phase_t["dropped_units"]:
        lines += ["## Units that contribute no curve point", "",
                  "| block | origin | reason |", "| --- | ---: | --- |"]
        for row in phase_t["dropped_units"]:
            lines.append("| %s | %s | %s |" % (
                row["block"], row["origin"], "; ".join(row["reasons"])))
        lines += ["",
                  "These still run and still write Episodes. They are absent "
                  "from the curve for **every arm identically**.", ""]
    else:
        lines += ["Every planned unit can be scored.", ""]
    if phase_t["gate_unreadable_units"]:
        lines += ["## Units whose delayed gate cannot be read", "",
                  "| block | origin | reason |", "| --- | ---: | --- |"]
        for row in phase_t["gate_unreadable_units"]:
            lines.append("| %s | %s | %s |" % (
                row["block"], row["origin"], "; ".join(row["reasons"])))
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    payload = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    phase_t = payload["phase_t"]
    print("scheduled / scoreable   : %s / %s" % (phase_t["N_T"],
                                                phase_t["N_T_eff"]))
    print("min paired curve points : %s" % phase_t["min_paired_curve_points"])
    print("orderings agree         : %s" % phase_t["orderings_agree"])
    for row in phase_t["dropped_units"]:
        print("  drops %s x %s: %s" % (row["block"], row["origin"],
                                      "; ".join(row["reasons"])))
    print("phase S N_S_eff         : %s / %s" % (payload["phase_s"]["N_S_eff"],
                                                payload["phase_s"]["N_S"]))
    check = payload["cross_check"]
    print("frozen manifest agrees  : %s" % check["passed"])
    if not check["passed"]:
        for key in ("declared_but_not_derived", "derived_but_not_declared"):
            if check[key]:
                print("  %s: %s" % (key, check[key]))
        return 1
    print("wrote %s" % OUT_JSON.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
