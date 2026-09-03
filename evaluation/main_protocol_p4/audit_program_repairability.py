"""Is there anything in the legal Program space for Slow to repair towards?

P4b closed on a fact, not a threshold: a program that looks safe on Support-A is
not safe on Support-B, and those two faces are **disjoint series splits at the
same origin** -- so what failed is cross-series transfer, not time.  Before
building a repair mechanism, the question has to be asked without an LLM: does
the existing legal space contain any variant that is stable on *both* faces?

The sweep is exhaustive over what a Slow Patch is allowed to change:

* **Workflow** -- every eligible single operator, and every ordered two-step
  composition of them.
* **Targeting** -- the operator's declared ``targeting_mode`` (``intrinsic`` vs
  ``global``), which is a property of the operator, so sweeping operators sweeps
  targeting.
* **Parameter strength** -- the public schema of every operator that has one.

Each variant is read on both faces at each held-in origin and scored against the
same bounded rule the experiment used.  No LLM, no Outcome, no held-out origin,
no new operator, and no threshold is changed -- the rule is imported, not
restated.

A variant is **A/B stable** at an origin when both faces clear all three of:
aggregate gain >= +0.005, harmed fraction <= 0.20, max single-series harm <= 0.30.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import p4b_viability as viability
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from SelfEvolvingHarnessTS.methods.ttha import admission_policy
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import classify_relation
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
from SelfEvolvingHarnessTS.operators.registry import operator_targeting_mode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT = PROJECT_ROOT / "artifacts/main_protocol/p4c_program_repairability_audit.json"

FACES = ("support_a", "support_b")
MATERIAL = 0.005

# Parameter sweeps, taken from each operator's public schema rather than
# invented: minimum, default, and a spread inside the declared bounds.  This is
# the "processing strength" axis a Slow Patch is allowed to move.
PARAMETER_GRID: Mapping[str, Mapping[str, Sequence[Any]]] = {
    "impute_linear": {"strength": (0.25, 0.5, 0.75, 1.0)},
    "denoise_median": {"window": (1, 3, 5, 7, 11), "strength": (0.25, 0.5, 1.0)},
    "hampel_filter": {
        "window": (3, 5, 7, 11, 25),
        "n_sigmas": (1.0, 2.0, 3.0, 4.5, 6.0),
        "global_z_min": (0.0,),
    },
    "period_median_complete": {
        "period": (forecast_p1.PERIOD,),
        "cycles": (1, 2, 3, 4),
        "min_donors": (1, 2, 3),
    },
}


def _param_variants(op: str) -> list[dict[str, Any]]:
    """Every parameter setting for one operator, defaults first."""
    default = dict(forecast_p1._params(op))
    grid = PARAMETER_GRID.get(op)
    if not grid:
        return [default]
    names = sorted(grid)
    variants = [default]
    for combination in itertools.product(*(grid[name] for name in names)):
        candidate = dict(default)
        candidate.update(dict(zip(names, combination)))
        if candidate not in variants:
            variants.append(candidate)
    return variants


def program_space(*, two_step: bool) -> list[dict[str, Any]]:
    """The legal Program space this audit exhausts."""
    ops = list(forecast_p1._eligible_programs())
    programs: list[dict[str, Any]] = []
    for op in ops:
        for params in _param_variants(op):
            programs.append(
                {
                    "program_id": "%s%s" % (
                        op,
                        "" if params == dict(forecast_p1._params(op))
                        else "[" + ",".join(
                            "%s=%s" % (k, params[k]) for k in sorted(params)
                        ) + "]",
                    ),
                    "steps": [{"op": op, "params": params}],
                    "workflow_length": 1,
                    "targeting_modes": [operator_targeting_mode(op)],
                }
            )
    if two_step:
        # Ordered pairs at default parameters: the Workflow axis on its own,
        # held apart from the strength axis so a finding names which one moved.
        for first, second in itertools.product(ops, ops):
            programs.append(
                {
                    "program_id": "%s>%s" % (first, second),
                    "steps": [
                        {"op": first, "params": dict(forecast_p1._params(first))},
                        {"op": second, "params": dict(forecast_p1._params(second))},
                    ],
                    "workflow_length": 2,
                    "targeting_modes": [
                        operator_targeting_mode(first),
                        operator_targeting_mode(second),
                    ],
                }
            )
    return programs


def _executor(cell: Any, face: str, origin: int, identity: Mapping[str, Any]) -> Any:
    """The same evaluator the online loop probes through.

    ``forecast_p4._reading`` is the *fixed comparator* and refuses any
    multi-step program by design, so using it here would silently skip the
    entire Workflow-composition axis.  ``ScopeExecutor`` is what the loop
    actually probes with: it compiles multi-step programs, and it applies the
    window verifier, so a composition the harness would refuse is reported as
    refused rather than as absent.
    """
    executor = ScopeExecutor(
        cell.roster(face),
        cell.values,
        forecast_p4._config(origin),
        evaluate_fn=forecast_p1.forecast_runtime._evaluate,
        max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION,
    )
    executor._baseline_cache[origin] = float(identity["smase"])
    executor._per_view_cache[origin] = [
        float(value) for value in identity["per_series_smase"]
    ]
    return executor


def _face_reading(
    executor: Any, steps: Sequence[Mapping[str, Any]], origin: int,
) -> dict[str, Any]:
    """One face's gain and risk profile, through the loop's own evaluator."""
    compiled = tuple((str(s["op"]), dict(s["params"])) for s in steps)
    try:
        receipt = executor.evaluate(compiled, origin)
    except Exception as exc:  # noqa: BLE001 - an illegal variant is simply absent
        return {"failed": "%s: %s" % (type(exc).__name__, str(exc)[:120])}
    if receipt.gain is None:
        return {"failed": str(receipt.error or "window_verifier_rejected")[:120]}
    gains = np.asarray(receipt.per_view_gain or (), dtype=np.float64)
    if gains.size == 0:
        return {"failed": "no per-series reading"}
    harmed = gains < -MATERIAL
    lowest = float(gains.min())
    return {
        "aggregate_gain": float(receipt.gain),
        "series_count": int(gains.size),
        "harmed_count": int(harmed.sum()),
        "harmed_fraction": float(harmed.mean()),
        "max_single_series_harm": -lowest if lowest < 0.0 else 0.0,
        "per_series_gain": [float(value) for value in gains],
    }


def _verdict(reading: Mapping[str, Any] | None, policy: Any) -> dict[str, Any]:
    """Score one face with the experiment's own rule, imported not restated.

    ``classify_relation`` supplies the relation exactly as the loop does, so a
    variant is judged here by the same two-step the live gate uses: classify,
    then admit under the active policy.
    """
    if reading is None or "failed" in (reading or {}):
        return {"admitted": False, "reason": "program_did_not_run"}
    per_series = {
        "s%d" % index: value
        for index, value in enumerate(reading["per_series_gain"])
    }
    facts = classify_relation(
        aggregate_gain=reading["aggregate_gain"],
        per_series_gains=per_series,
        material_threshold=MATERIAL,
    )
    verdict = admission_policy.decide(
        relation=str(facts["relation"]),
        aggregate_gain=reading["aggregate_gain"],
        per_series_gains=per_series,
        policy=policy,
    ).to_dict()
    verdict["relation"] = str(facts["relation"])
    return verdict


def audit(base_cell: Any, origins: Sequence[int], programs: Sequence[Mapping[str, Any]],
          *, progress_every: int = 25) -> dict[str, Any]:
    started = time.time()
    rows: list[dict[str, Any]] = []
    identity_by_origin: dict[int, dict[str, Any]] = {}
    for origin in origins:
        cell = forecast_p4._cell_at(base_cell, int(origin))
        identity_by_origin[int(origin)] = {
            face: forecast_p4._reading(cell, face, (), origin=int(origin))
            for face in FACES
        }
    fits = len(origins) * len(FACES)

    # One executor per (origin, face), reused across programs: building it is
    # roster bookkeeping, and the identity baseline is already known.
    executors = {
        (int(origin), face): _executor(
            forecast_p4._cell_at(base_cell, int(origin)), face, int(origin),
            identity_by_origin[int(origin)][face],
        )
        for origin in origins
        for face in FACES
    }

    for index, program in enumerate(programs, start=1):
        per_origin = []
        for origin in origins:
            faces = {}
            for face in FACES:
                reading = _face_reading(
                    executors[(int(origin), face)], program["steps"], int(origin)
                )
                fits += 1
                faces[face] = reading
            verdicts = {
                face: _verdict(faces[face], contract.BOUNDED_POLICY)
                for face in FACES
            }
            stable = all(verdicts[face].get("admitted") for face in FACES)
            per_origin.append(
                {
                    "origin": int(origin),
                    "stable_on_both_faces": bool(stable),
                    **{
                        face: {
                            key: faces[face].get(key)
                            for key in ("aggregate_gain", "harmed_fraction",
                                        "max_single_series_harm", "harmed_count",
                                        "failed")
                            if faces[face].get(key) is not None
                        }
                        for face in FACES
                    },
                    "verdicts": {
                        face: {
                            "admitted": verdicts[face].get("admitted"),
                            "relation": verdicts[face].get("relation"),
                            "reason": verdicts[face].get("reason"),
                        }
                        for face in FACES
                    },
                }
            )
        stable_origins = [
            entry["origin"] for entry in per_origin if entry["stable_on_both_faces"]
        ]
        unreadable = [
            entry["origin"] for entry in per_origin
            if any("failed" in entry[face] for face in FACES)
        ]
        rows.append(
            {
                "program_id": program["program_id"],
                "steps": program["steps"],
                "workflow_length": program["workflow_length"],
                "targeting_modes": program["targeting_modes"],
                "stable_origin_count": len(stable_origins),
                "stable_origins": stable_origins,
                "unreadable_origins": unreadable,
                "readable_origin_count": len(per_origin) - len(unreadable),
                "a_face_admitted_origins": [
                    entry["origin"] for entry in per_origin
                    if entry["verdicts"]["support_a"]["admitted"]
                ],
                "b_face_admitted_origins": [
                    entry["origin"] for entry in per_origin
                    if entry["verdicts"]["support_b"]["admitted"]
                ],
                "per_origin": per_origin,
            }
        )
        if index % progress_every == 0:
            print("  %d/%d programs  (%.1f min)" % (
                index, len(programs), (time.time() - started) / 60), flush=True)
    return {"rows": rows, "consumer_fits": fits,
            "wall_seconds": round(time.time() - started, 1)}


NULL_EFFECT = 1e-9


def _breach(face: Mapping[str, Any]) -> tuple[float, float, float] | None:
    """How far one face is from clearing, per condition.  All zero = cleared."""
    if "aggregate_gain" not in face:
        return None
    return (
        max(0.0, MATERIAL - face["aggregate_gain"]),
        max(0.0, face.get("harmed_fraction", 1.0)
            - contract.BOUNDED_MAX_HARMED_FRACTION),
        max(0.0, face.get("max_single_series_harm", 1.0)
            - contract.BOUNDED_MAX_SINGLE_SERIES_HARM),
    )


def near_misses(rows: Sequence[Mapping[str, Any]], limit: int = 10) -> dict[str, Any]:
    """The effective programs that came closest to clearing both faces.

    "Closest" has to exclude null programs: a variant that changes nothing
    reads as gain 0.0 with no harm, which scores as a small breach while being
    identity under another name.  What matters is how far a program that
    *actually acts* is from being deployable, because that is the distance a
    Slow Patch would have to close.
    """
    scored = []
    nulls = 0
    for row in rows:
        for entry in row["per_origin"]:
            face_a, face_b = entry["support_a"], entry["support_b"]
            breach_a, breach_b = _breach(face_a), _breach(face_b)
            if breach_a is None or breach_b is None:
                continue
            if (abs(face_a["aggregate_gain"]) < NULL_EFFECT
                    and abs(face_b["aggregate_gain"]) < NULL_EFFECT):
                nulls += 1
                continue
            scored.append(
                {
                    "total_breach": sum(breach_a) + sum(breach_b),
                    "program_id": row["program_id"],
                    "origin": entry["origin"],
                    "support_a": {k: face_a.get(k) for k in (
                        "aggregate_gain", "harmed_fraction",
                        "max_single_series_harm", "harmed_count")},
                    "support_b": {k: face_b.get(k) for k in (
                        "aggregate_gain", "harmed_fraction",
                        "max_single_series_harm", "harmed_count")},
                    "breach_support_a": dict(zip(
                        ("gain", "harmed_fraction", "max_single_series_harm"),
                        breach_a)),
                    "breach_support_b": dict(zip(
                        ("gain", "harmed_fraction", "max_single_series_harm"),
                        breach_b)),
                }
            )
    scored.sort(key=lambda entry: entry["total_breach"])
    return {
        "null_effect_cells_excluded": nulls,
        "closest": scored[:limit],
        "smallest_total_breach": scored[0]["total_breach"] if scored else None,
    }


def face_clearance(rows: Sequence[Mapping[str, Any]],
                   origins: Sequence[int]) -> list[dict[str, Any]]:
    """Per origin, how many programs clear each face -- and how many clear both."""
    table = []
    for origin in origins:
        table.append(
            {
                "origin": int(origin),
                "support_a_cleared_by": sum(
                    1 for row in rows if origin in row["a_face_admitted_origins"]),
                "support_b_cleared_by": sum(
                    1 for row in rows if origin in row["b_face_admitted_origins"]),
                "both_faces_cleared_by": sum(
                    1 for row in rows if origin in row["stable_origins"]),
            }
        )
    return table


def summarise(rows: Sequence[Mapping[str, Any]], origins: Sequence[int]) -> dict[str, Any]:
    """Is there headroom, and if so where."""
    stable_anywhere = [row for row in rows if row["stable_origin_count"] > 0]
    a_only = [
        row for row in rows
        if row["a_face_admitted_origins"] and not row["b_face_admitted_origins"]
    ]
    best = sorted(rows, key=lambda row: -row["stable_origin_count"])[:15]
    readable = [row for row in rows if row["readable_origin_count"] > 0]
    return {
        "programs_swept": len(rows),
        "programs_readable_somewhere": len(readable),
        "programs_never_readable": len(rows) - len(readable),
        "origins": list(origins),
        "programs_stable_on_both_faces_somewhere": len(stable_anywhere),
        "programs_stable_at_every_origin": sum(
            1 for row in rows if row["stable_origin_count"] == len(origins)
        ),
        "programs_admitted_on_a_but_never_on_b": len(a_only),
        "headroom_exists": bool(stable_anywhere),
        "best_programs": [
            {
                "program_id": row["program_id"],
                "stable_origins": row["stable_origins"],
                "stable_origin_count": row["stable_origin_count"],
                "workflow_length": row["workflow_length"],
                "targeting_modes": row["targeting_modes"],
            }
            for row in best
        ],
        "face_clearance_by_origin": face_clearance(rows, origins),
        "near_misses": near_misses(rows),
        "by_axis": {
            "single_step_stable": sum(
                1 for row in rows
                if row["workflow_length"] == 1 and row["stable_origin_count"] > 0
            ),
            "two_step_readable": sum(
                1 for row in rows
                if row["workflow_length"] == 2 and row["readable_origin_count"] > 0
            ),
            "two_step_stable": sum(
                1 for row in rows
                if row["workflow_length"] == 2 and row["stable_origin_count"] > 0
            ),
            "single_step_readable": sum(
                1 for row in rows
                if row["workflow_length"] == 1 and row["readable_origin_count"] > 0
            ),
            "parameter_variant_stable": sum(
                1 for row in rows
                if "[" in row["program_id"] and row["stable_origin_count"] > 0
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-two-step", action="store_true",
                        help="sweep single operators and parameters only")
    parser.add_argument("--origins", type=int, default=None,
                        help="use only the first N held-in origins (smoke only)")
    args = parser.parse_args(argv)

    base_cell, _selection, data = forecast_p1._load_exposed_cells()
    plan = contract.resolve_origins(viability.screen(base_cell))
    origins = list(plan["held_in_origins"])[: args.origins]
    programs = program_space(two_step=not args.no_two_step)
    print("sweeping %d programs x %d origins x 2 faces" % (len(programs), len(origins)))

    result = audit(base_cell, origins, programs)
    summary = summarise(result["rows"], origins)
    report = {
        "stage": "P4C_PROGRAM_REPAIRABILITY_AUDIT",
        "written_at": datetime.now().astimezone().isoformat(),
        "question": (
            "does the legal Program space contain any variant that clears the "
            "bounded rule on both Support-A and Support-B at the same origin?"
        ),
        "why_it_comes_first": (
            "P4b closed with 9 Support-A admissions and 0 approvals; before "
            "building a Slow repair mechanism, the space it would repair towards "
            "has to be shown non-empty"
        ),
        "faces": {
            "definition": (
                "Support-A and Support-B are disjoint series splits at the same "
                "origin (20 eval series each, zero overlap), so A/B stability is "
                "cross-series transfer, not temporal stability"
            ),
        },
        "rule": {
            "source": "methods/ttha/admission_policy.py BOUNDED_V1, imported",
            "max_harmed_fraction": contract.BOUNDED_MAX_HARMED_FRACTION,
            "max_single_series_harm": contract.BOUNDED_MAX_SINGLE_SERIES_HARM,
            "material_line": MATERIAL,
            "thresholds_changed": False,
        },
        "boundary": {
            "llm_calls": 0,
            "held_out_origins_touched": 0,
            "new_operators_introduced": 0,
            "data_role": "EXPOSED_HELD_IN_ONLY",
        },
        "dataset": data.get("dataset"),
        "consumer_fits": result["consumer_fits"],
        "wall_seconds": result["wall_seconds"],
        "summary": summary,
        "rows": result["rows"],
        "verdict": (
            "REPAIR_HEADROOM_EXISTS" if summary["headroom_exists"]
            else "NO_REPAIR_HEADROOM"
        ),
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("\nprograms swept              : %d" % summary["programs_swept"])
    print("stable on both faces (any)  : %d" % summary["programs_stable_on_both_faces_somewhere"])
    print("stable at every origin      : %d" % summary["programs_stable_at_every_origin"])
    print("admitted on A but never B   : %d" % summary["programs_admitted_on_a_but_never_on_b"])
    print("readable somewhere          : %d  (never readable: %d)" % (
        summary["programs_readable_somewhere"], summary["programs_never_readable"]))
    print("   single-step readable/stable: %d / %d" % (
        summary["by_axis"]["single_step_readable"], summary["by_axis"]["single_step_stable"]))
    print("   two-step   readable/stable: %d / %d" % (
        summary["by_axis"]["two_step_readable"], summary["by_axis"]["two_step_stable"]))
    print("consumer fits               : %d  (0 LLM)" % result["consumer_fits"])
    print("verdict                     : %s" % report["verdict"])
    for entry in summary["best_programs"][:8]:
        print("   %-40s stable at %d origins %s" % (
            entry["program_id"], entry["stable_origin_count"], entry["stable_origins"]))
    print("wrote %s" % REPORT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
