"""Can the serving-side scoped pipeline be frozen, before any Harness run?

Seven gates.  None measures utility; the experiment that does is not authorised
until this contract and the ScopeSpec above it are frozen.

1. **Static equivalence.**  With no Program the new evaluator must reproduce the
   frozen path to the bit, or every later reading is confounded.
2. **train_only equivalence.**  With ``serving_mode="train_only"`` and a real
   Program it must reproduce the frozen path exactly, so P4D and P4M stay
   reproducible from this module and the two lines remain comparable.
3. **Empty scope is Static.**  Declining to treat anything must give literally
   the Static numbers -- this is what makes abstention a real action rather than
   a relabelled treatment.
4. **Unscoped series are untouched.**  Under a partial scope, every series the
   Harness declined must score bit-identically to Static, and at least one
   scoped series must move.  This is the property the training-row scope could
   not provide: ``p4n`` measured 20/20 evaluation series moving when a single
   training series was scoped.
5. **Causality.**  Perturbing the series strictly after the scored horizon must
   change nothing; the served context is ``raw[origin-192:origin]`` and the
   metric scale reads ``raw[:origin]``.
6. **Billing.**  Two pipelines cost two Consumer fits, and the count must be
   reported rather than hidden.
7. **Classification parity.**  The same three-surface semantics must exist on
   the Classification adapter.  It does not yet, and the gate says so instead of
   passing quietly.

0 LLM calls, no held-out read, no frozen path modified.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from evaluation.main_protocol_p4 import audit_gap_repairability as gaps
from evaluation.main_protocol_p4 import audit_param_correction_rerun as fixes
from evaluation.main_protocol_p4 import audit_program_repairability as p4c
from evaluation.main_protocol_p4 import phase2_contract as contract
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4o_scoped_serving_preflight.json"

FACES = contract.FACES
PROBE = ("period_median_complete", "winsorize")


def _steps() -> tuple[tuple[str, dict[str, Any]], ...]:
    return tuple(
        (str(step["op"]), dict(step["params"]))
        for step in fixes.corrected_steps(
            [{"op": op, "params": forecast_p1._params(op)} for op in PROBE]
        )
    )


def _context(cell: Any, origin: int, face: str) -> tuple[Any, Any, Any]:
    at = forecast_p4._cell_at(cell, int(origin))
    config = forecast_p4._config(int(origin))
    roster = at.roster(face)
    executor = p4c.ScopeExecutor(
        roster, at.values, config,
        evaluate_fn=forecast_runtime._evaluate,
        max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION,
    )
    return at, config, (roster, executor)


def _bitwise(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def run_gates(cell: Any, origin: int) -> dict[str, Any]:
    steps = _steps()
    rows: dict[str, list[dict[str, Any]]] = {
        "static": [], "train_only": [], "empty_scope": [],
        "partial_scope": [], "causality": [], "billing": [],
    }
    for face in FACES:
        at, config, (roster, executor) = _context(cell, origin, face)
        eval_uids = [
            str(row["series_uid"]) for row in roster if row["role"] == "eval"
        ]
        frozen_static = forecast_runtime._evaluate(
            roster, at.values, None, config, origin=int(origin))
        mine_static = scoped.scoped_evaluate(
            roster, at.values, None, config, origin=int(origin))
        rows["static"].append({
            "face": face,
            "max_difference": _bitwise(
                frozen_static["per_view_smase"], mine_static["per_view_smase"]),
            "consumer_fits": mine_static["consumer_fits"],
        })
        if not executor.verify(steps, int(origin)).passed:
            rows["train_only"].append({"face": face, "skipped": "verifier refused"})
            continue
        compiled = executor._compiled(steps)

        frozen_program = forecast_runtime._evaluate(
            roster, at.values, compiled, config, origin=int(origin))
        train_only = scoped.scoped_evaluate(
            roster, at.values, compiled, config, origin=int(origin),
            serving_mode="train_only")
        rows["train_only"].append({
            "face": face,
            "max_difference": _bitwise(
                frozen_program["per_view_smase"], train_only["per_view_smase"]),
            "behavior_matches": (
                int(frozen_program["behavior_point_count"])
                == int(train_only["behavior_point_count"])),
        })

        empty = scoped.scoped_evaluate(
            roster, at.values, compiled, config, origin=int(origin),
            scope=frozenset())
        rows["empty_scope"].append({
            "face": face,
            "max_difference_vs_static": _bitwise(
                frozen_static["per_view_smase"], empty["per_view_smase"]),
            "program_pipeline_used": empty["program_pipeline_used"],
            "consumer_fits": empty["consumer_fits"],
        })

        half = frozenset(eval_uids[:len(eval_uids) // 2])
        partial = scoped.scoped_evaluate(
            roster, at.values, compiled, config, origin=int(origin), scope=half)
        losses = np.asarray(partial["per_view_smase"], dtype=np.float64)
        static = np.asarray(frozen_static["per_view_smase"], dtype=np.float64)
        selected = np.array([uid in half for uid in eval_uids])
        rows["partial_scope"].append({
            "face": face,
            "scope_size": int(selected.sum()),
            "unscoped_max_difference_vs_static": float(
                np.max(np.abs(losses[~selected] - static[~selected]))),
            "scoped_series_that_moved": int(np.count_nonzero(
                ~np.isclose(losses[selected], static[selected], atol=1e-12))),
            "scoped_series": int(selected.sum()),
        })
        rows["billing"].append({
            "face": face,
            "no_program": mine_static["consumer_fits"],
            "empty_scope": empty["consumer_fits"],
            "partial_scope": partial["consumer_fits"],
        })

        # Strictly-future perturbation: nothing beyond the scored horizon may
        # reach a reading.
        polluted = dict(at.values)
        tail = int(origin) + scoped.HORIZON
        for uid in eval_uids:
            series = np.array(at.values[uid], dtype=np.float64)
            if series.size > tail:
                series[tail:] = 1e6
            polluted[uid] = series
        after = scoped.scoped_evaluate(
            roster, polluted, compiled, config, origin=int(origin), scope=half)
        rows["causality"].append({
            "face": face,
            "max_difference_after_future_perturbation": _bitwise(
                partial["per_view_smase"], after["per_view_smase"]),
        })
    return rows


def build() -> dict[str, Any]:
    support_a, support_b, _origins = gaps._roster_from_preflight()
    cell = gaps._variant_cell(support_a, support_b)
    origin = int(contract.ORIGINS[0])
    rows = run_gates(cell, origin)

    def _max(key: str, field: str) -> float:
        values = [row[field] for row in rows[key] if field in row]
        return max(values) if values else float("nan")

    gates = {
        "gate_1_static_equivalence": {
            "rows": rows["static"],
            "max_difference": _max("static", "max_difference"),
            "passed": _max("static", "max_difference") == 0.0,
            "reading": "no Program: the new evaluator is the frozen path",
        },
        "gate_2_train_only_equivalence": {
            "rows": rows["train_only"],
            "max_difference": _max("train_only", "max_difference"),
            "passed": (
                _max("train_only", "max_difference") == 0.0
                and all(row.get("behavior_matches", False)
                        for row in rows["train_only"] if "skipped" not in row)
            ),
            "reading": "train_only reproduces P4D/P4M exactly",
        },
        "gate_3_empty_scope_is_static": {
            "rows": rows["empty_scope"],
            "passed": (
                _max("empty_scope", "max_difference_vs_static") == 0.0
                and not any(row["program_pipeline_used"]
                            for row in rows["empty_scope"])
            ),
            "reading": "declining to treat anything gives literally Static",
        },
        "gate_4_unscoped_series_are_untouched": {
            "rows": rows["partial_scope"],
            "passed": (
                _max("partial_scope", "unscoped_max_difference_vs_static") == 0.0
                and all(row["scoped_series_that_moved"] > 0
                        for row in rows["partial_scope"])
            ),
            "reading": (
                "under a partial scope the declined series are bit-identical to "
                "Static and the treated series move -- the property a "
                "training-row scope could not provide"
            ),
        },
        "gate_5_causality": {
            "rows": rows["causality"],
            "passed": _max(
                "causality", "max_difference_after_future_perturbation") == 0.0,
            "reading": "nothing beyond the scored horizon reaches a reading",
        },
        "gate_6_billing": {
            "rows": rows["billing"],
            "passed": all(
                row["no_program"] == 1 and row["empty_scope"] == 1
                and row["partial_scope"] == 2 for row in rows["billing"]),
            "reading": "two pipelines cost two Consumer fits and are reported",
        },
        "gate_7_classification_parity": {
            "passed": False,
            "status": "NOT_YET_IMPLEMENTED",
            "required_change": (
                "MacroF1ConsumerAdapter must prepare cell.surface(face) with the "
                "same steps it applies in _prepared_fit, keep labels raw, and "
                "fit a second raw model so unscoped instances are bit-identical "
                "to Static"
            ),
            "reading": (
                "the Forecast side is ready; Classification still serves "
                "unprepared features and has no raw fallback pipeline"
            ),
        },
    }
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    return {
        "stage": "P4O_SCOPED_SERVING_PREFLIGHT",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_CONTRACT_CHECK",
        "data_version": contract.DATA_VERSION,
        "origin_probed": origin,
        "boundary": {
            "llm_calls": 0,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "frozen_evaluator_modified": False,
            "harness_runs": 0,
        },
        "surfaces": {
            "train_context_and_target": "prepared, as the frozen path did",
            "serve_context": "prepared causally from raw[origin-192:origin]",
            "evaluation_truth": "always raw, missing-aware sMASE on 48 steps",
        },
        "gates": gates,
        "failed_gates": failed,
        "verdict": (
            "FORECAST_SCOPED_SERVING_READY" if failed == [
                "gate_7_classification_parity"]
            else "SCOPED_SERVING_READY" if not failed
            else "SCOPED_SERVING_BLOCKED"
        ),
        "releases": (
            "the Forecast serving contract only; ScopeSpec and the lifecycle "
            "preflight are still required before any Harness run"
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
        print("%-36s %-6s %s" % (
            name.replace("gate_", "").replace("_", " ")[:36],
            "PASS" if gate["passed"] else "BLOCK", gate["reading"]))
    print("verdict : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
