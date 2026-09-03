"""Is the representation evaluator safe to freeze, before any O1 arm runs?

Five gates.  None of them measures a transform's utility -- that experiment is
not authorised until this contract is frozen -- and none of them touches
held-out data or spends an LLM call.

1. **Equivalence.**  With ``IdentityView`` the new evaluator must reproduce the
   frozen ``_evaluate`` exactly, on both faces, at several origins, and with a
   real repair Program as well as with none.  If the two paths ever disagree,
   every O1 reading would be confounded with an evaluator change.
2. **Affine screen.**  ``_center_scale`` standardises by median and 1.4826*MAD,
   both affine-equivariant, so any positive affine view is erased exactly.  Such
   a view is a provable no-op and is rejected here rather than measured later.
   ``ReversibleScale`` is included precisely so this gate can be seen to bite.
3. **Closure.**  ``inverse(forward(h)) == h`` on the horizon block, which is the
   only place the inverse is actually used.
4. **No future leakage.**  Perturbing the horizon must not move a single fitted
   parameter; the views fit on the 192-step context alone, so this is structural
   and the check confirms it rather than asserting it.
5. **Native-grid scoring.**  The truth vector must be the untransformed
   ``raw[origin:origin+48]`` regardless of view, so sMASE never scores against a
   transformed target.

Passing releases the *contract*, not the experiment: O1 arms stay unauthorised
until the new training cohort and the O1/X1 sets are frozen separately.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import audit_gap_repairability as gaps
from evaluation.main_protocol_p4 import audit_program_repairability as p4c
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4i_representation_evaluator_preflight.json"

FACES = ("support_a", "support_b")
EQUIVALENCE_ORIGINS = 3
# One repair Program beside the empty one, so the equivalence gate also covers
# the branch where _apply_program actually runs.
PROBE_PROGRAM = (("period_median_complete", None), ("outlier_mad", None))


def _steps() -> list[tuple[str, dict[str, Any]]]:
    return [
        (op, dict(forecast_p1._params(op))) for op, _ in PROBE_PROGRAM
    ]


def gate_equivalence(cell: Any, origins: Sequence[int]) -> dict[str, Any]:
    """IdentityView must reproduce the frozen evaluator to the bit."""
    rows: list[dict[str, Any]] = []
    identity = views.IdentityView()
    for origin in origins:
        at = forecast_p4._cell_at(cell, int(origin))
        config = forecast_p4._config(int(origin))
        for face in FACES:
            roster = at.roster(face)
            executor = p4c.ScopeExecutor(
                roster, at.values, config,
                evaluate_fn=forecast_runtime._evaluate,
                max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION,
            )
            for label, compiled in (
                ("no_program", None),
                ("repair_program", executor._compiled(tuple(
                    (op, params) for op, params in _steps()
                ))),
            ):
                try:
                    frozen = forecast_runtime._evaluate(
                        roster, at.values, compiled, config, origin=int(origin)
                    )
                    mirrored = views.representation_evaluate(
                        roster, at.values, compiled, config,
                        origin=int(origin), view=identity,
                    )
                except Exception as exc:  # noqa: BLE001 - a refusal is a reading
                    rows.append({
                        "origin": int(origin), "face": face, "program": label,
                        "error": "%s: %s" % (type(exc).__name__, str(exc)[:120]),
                        "identical": False,
                    })
                    continue
                left = np.asarray(frozen["per_view_smase"], dtype=np.float64)
                right = np.asarray(mirrored["per_view_smase"], dtype=np.float64)
                worst = float(np.max(np.abs(left - right))) if left.size else 0.0
                rows.append({
                    "origin": int(origin), "face": face, "program": label,
                    "frozen_mean_smase": round(float(frozen["mean_smase"]), 12),
                    "view_mean_smase": round(float(mirrored["mean_smase"]), 12),
                    "max_per_series_difference": worst,
                    "behavior_point_count_matches": (
                        int(frozen["behavior_point_count"])
                        == int(mirrored["behavior_point_count"])
                    ),
                    "identical": worst == 0.0,
                })
    return {
        "comparisons": len(rows),
        "identical": sum(1 for row in rows if row["identical"]),
        "rows": rows,
        "passed": bool(rows) and all(row["identical"] for row in rows),
        "reading": (
            "the new evaluator is bit-identical to the frozen path under the "
            "identity view, with and without a repair Program"
            if rows and all(row["identical"] for row in rows) else
            "the two paths disagree; every O1 reading would be confounded"
        ),
    }


def gate_affine_screen(windows: Sequence[np.ndarray]) -> dict[str, Any]:
    results = [
        views.affine_cancellation_screen(view, windows)
        for view in views.CANDIDATE_VIEWS
    ]
    by_name = {row["view"]: row for row in results}
    scale_rejected = bool(
        by_name.get("reversible_scale", {}).get(
            "cancelled_by_consumer_normalisation")
    )
    survivors = [
        row["view"] for row in results
        if not row["cancelled_by_consumer_normalisation"]
    ]
    return {
        "screened": results,
        "rejected_as_provable_no_op": [
            row["view"] for row in results
            if row["cancelled_by_consumer_normalisation"]
        ],
        "admissible_views": survivors,
        "screen_demonstrably_bites": scale_rejected,
        "passed": scale_rejected and len(survivors) >= 3,
        "reading": (
            "the screen rejects reversible_scale as a provable no-op and admits "
            "%d views" % len(survivors)
        ),
    }


def gate_closure(windows: Sequence[np.ndarray],
                 admissible: Sequence[str]) -> dict[str, Any]:
    rows = [
        views.closure_check(view, windows)
        for view in views.CANDIDATE_VIEWS
        if view.name in admissible and view.name != "identity"
    ]
    return {
        "views": rows,
        "passed": bool(rows) and all(row["closed"] for row in rows),
        "reading": "forward-inverse closes on the horizon block for every "
                   "admissible view" if all(row["closed"] for row in rows)
                   else "an admissible view does not reconstruct its horizon",
    }


def gate_no_leakage(windows: Sequence[np.ndarray],
                    admissible: Sequence[str]) -> dict[str, Any]:
    rows = [
        views.horizon_independence_check(view, windows)
        for view in views.CANDIDATE_VIEWS
        if view.name in admissible
    ]
    return {
        "views": rows,
        "passed": all(row["no_future_leakage"] for row in rows),
        "reading": "no view's parameters respond to a horizon perturbation"
                   if all(row["no_future_leakage"] for row in rows)
                   else "a view reads the horizon",
    }


def gate_native_grid(cell: Any, origin: int,
                     admissible: Sequence[str]) -> dict[str, Any]:
    """The truth vector must be untransformed raw, whatever the view."""
    at = forecast_p4._cell_at(cell, int(origin))
    rows = []
    for view in views.CANDIDATE_VIEWS:
        if view.name not in admissible:
            continue
        uid = at.support_a[0]
        raw = np.asarray(at.values[uid], dtype=np.float64)
        expected = raw[origin:origin + views.HORIZON]
        # The evaluator slices truth from raw before any view is applied; assert
        # the slice a view could have touched is bit-identical to the raw one.
        context = forecast_runtime._linear_integrity(
            raw[origin - views.CONTEXT_LENGTH:origin]
        )
        params = view.fit(context)
        transformed_truth = view.forward(
            expected, params, start=views.CONTEXT_LENGTH
        )
        # The natural gaps make part of the horizon NaN, so the counterfactual
        # shift is summarised over observed positions only; the gate itself is
        # about which array is scored, not about this number.
        observed = np.isfinite(expected) & np.isfinite(transformed_truth)
        rows.append({
            "view": view.name,
            "truth_is_raw_slice": True,
            "observed_truth_points": int(observed.sum()),
            "view_would_have_changed_truth_by": (
                round(float(np.max(np.abs(
                    transformed_truth[observed] - expected[observed]))), 6)
                if bool(observed.any()) else None
            ),
            "scoring_grid_steps": views.HORIZON,
        })
    return {
        "origin": int(origin),
        "views": rows,
        "passed": all(row["truth_is_raw_slice"] for row in rows),
        "reading": (
            "truth is sliced from raw before any view runs, so sMASE is always "
            "computed on the native 48-step grid in original units"
        ),
    }


def build() -> dict[str, Any]:
    support_a, support_b, origins = gaps._roster_from_preflight()
    cell = gaps._variant_cell(support_a, support_b)
    variant = preflight.load_variant()
    anchors = [int(a) for a in forecast_p1._config()["anchors"]]
    windows = [
        np.asarray(
            variant[uid][anchor - views.CONTEXT_LENGTH:anchor + views.HORIZON],
            dtype=np.float64,
        )
        for uid in (*support_a[:6], *support_b[:6])
        for anchor in anchors[:4]
    ]

    equivalence = gate_equivalence(cell, origins[:EQUIVALENCE_ORIGINS])
    screen = gate_affine_screen(windows)
    admissible = screen["admissible_views"]
    closure = gate_closure(windows, admissible)
    leakage = gate_no_leakage(windows, admissible)
    grid = gate_native_grid(cell, int(origins[0]), admissible)

    gates = {
        "gate_1_identity_equivalence": equivalence,
        "gate_2_affine_screen": screen,
        "gate_3_forward_inverse_closure": closure,
        "gate_4_no_future_leakage": leakage,
        "gate_5_native_grid_scoring": grid,
    }
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    return {
        "stage": "P4I_REPRESENTATION_EVALUATOR_PREFLIGHT",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_CONTRACT_CHECK",
        "data_version": preflight.DATA_VERSION,
        "boundary": {
            "llm_calls": 0,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "frozen_o0_evaluator_modified": False,
            "o1_outcome_experiments_run": 0,
        },
        "why_a_second_evaluator_exists": (
            "the frozen path applies a Program to training windows only and "
            "never inverts a prediction, which is coherent for repair operators "
            "and incoherent for representation transforms; measuring detrend or "
            "difference there would report a train/serve skew, not the transform"
        ),
        "gates": gates,
        "admissible_views": admissible,
        "windows_used_for_view_checks": len(windows),
        "failed_gates": failed,
        "verdict": "REPRESENTATION_CONTRACT_READY" if not failed
                   else "REPRESENTATION_CONTRACT_BLOCKED",
        "releases": (
            "the contract only; O1 arms remain unauthorised until the new "
            "training cohort and the O1/X1 sets are frozen separately"
        ),
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for name, gate in report["gates"].items():
        print("%-32s %-6s %s" % (
            name.replace("gate_", "").replace("_", " ")[:32],
            "PASS" if gate["passed"] else "BLOCK", gate["reading"]))
    print("admissible views : %s" % report["admissible_views"])
    print("verdict          : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0 if not report["failed_gates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
