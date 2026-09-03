"""Three Forecast operators were run without the period they need.

``run_forecast_p1._params`` returns ``{}`` for ``period_complete``, ``impute_ar``
and ``repair_level_shift``; the Classification component passes ``period=24`` to
exactly those three (``classification_component.py:548-549``).  The consequences
are not cosmetic:

* ``period_complete`` (``s1_impute.py:105-109``) returns ``interp_nan(y)``
  whenever ``period < 2``, so at ``period=0`` it *is* linear interpolation and
  therefore identical to identity.  Its measured ZERO_BEHAVIOR on 62 gapped
  windows was an instrument reading, not a null result.
* ``impute_ar`` derives its order as ``max(8, period)``; the operator's own
  docstring records that below the seasonal lag it cannot see the period at all
  (linear 2.00, AR(8) 1.33, AR(24) 0.34).  At ``period=0`` the Forecast line ran
  AR(8).
* ``repair_level_shift`` accepts ``period`` but declares no public property for
  it, so the correction is applied for consistency and reported separately.

The frozen P1 Common DSL is **not** edited: P1-P4c all ran against it and
changing it would retroactively alter what that contract meant.  The correction
is declared here and applies only to
``EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING``.  Only programs that touch
a corrected operator are rerun, and the uncorrected sweep is left in place.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import audit_gap_repairability as gaps
from evaluation.main_protocol_p4 import audit_program_repairability as p4c
from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4d_param_correction_rerun.json"
TENSOR = PROJECT_ROOT / "artifacts/main_protocol/p4d_param_corrected_gain.npz"
UNCORRECTED = PROJECT_ROOT / "artifacts/main_protocol/p4d_gap_per_series_gain.npz"

FACES = gaps.FACES
MATERIAL = gaps.MATERIAL

# The correction, stated as data so the artifact can carry it verbatim.
PARAM_CORRECTIONS: Mapping[str, dict[str, Any]] = {
    "period_complete": {"period": forecast_p1.PERIOD},
    "impute_ar": {"period": forecast_p1.PERIOD},
    "repair_level_shift": {"period": forecast_p1.PERIOD},
}


def corrected_steps(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for step in steps:
        op = str(step["op"])
        params = dict(step["params"])
        params.update(PARAM_CORRECTIONS.get(op, {}))
        out.append({"op": op, "params": params})
    return out


def affected(programs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        program for program in programs
        if any(str(step["op"]) in PARAM_CORRECTIONS for step in program["steps"])
    ]


def _admits(vector: np.ndarray) -> bool | None:
    if vector.size == 0 or bool(np.isnan(vector).any()):
        return None
    worst = float(-vector.min())
    return bool(
        float(vector.mean()) >= MATERIAL
        and float((vector < -MATERIAL).mean()) <= contract.BOUNDED_MAX_HARMED_FRACTION
        and max(worst, 0.0) <= contract.BOUNDED_MAX_SINGLE_SERIES_HARM
    )


def build() -> dict[str, Any]:
    started = time.time()
    support_a, support_b, origins = gaps._roster_from_preflight()
    cell = gaps._variant_cell(support_a, support_b)
    programs = affected(p4c.program_space(two_step=True))
    print("affected programs: %d | origins %s" % (len(programs), origins), flush=True)

    identity = {
        int(origin): {
            face: forecast_p4._reading(
                forecast_p4._cell_at(cell, int(origin)), face, (), origin=int(origin)
            )
            for face in FACES
        }
        for origin in origins
    }
    executors = {
        (int(origin), face): p4c._executor(
            forecast_p4._cell_at(cell, int(origin)), face, int(origin),
            identity[int(origin)][face],
        )
        for origin in origins
        for face in FACES
    }

    prior = np.load(UNCORRECTED, allow_pickle=True)
    prior_ids = [str(value) for value in prior["program_ids"]]
    prior_origins = [int(value) for value in prior["origins"]]
    prior_gain = prior["gain"]

    tensor = np.full(
        (len(programs), len(origins), len(FACES), len(support_a)), np.nan,
        dtype=np.float64,
    )
    rows, fits = [], len(origins) * len(FACES)
    for index, program in enumerate(programs):
        steps = corrected_steps(program["steps"])
        changed_faces, newly_stable = [], []
        for o_index, origin in enumerate(origins):
            vectors = {}
            for f_index, face in enumerate(FACES):
                reading = p4c._face_reading(
                    executors[(int(origin), face)], steps, int(origin)
                )
                fits += 1
                gains = reading.get("per_series_gain")
                if gains is not None and len(gains) == len(support_a):
                    tensor[index, o_index, f_index, :] = gains
                    vectors[face] = np.asarray(gains, dtype=np.float64)
            before = None
            if program["program_id"] in prior_ids:
                p_index = prior_ids.index(program["program_id"])
                po = prior_origins.index(int(origin))
                before = {
                    face: prior_gain[p_index, po, f_index, :]
                    for f_index, face in enumerate(FACES)
                }
            after_stable = all(
                _admits(vectors.get(face, np.array([]))) for face in FACES
            )
            before_stable = (
                None if before is None else
                all(_admits(before[face]) for face in FACES)
            )
            if after_stable and not before_stable:
                newly_stable.append(int(origin))
            if before is not None:
                for face in FACES:
                    now = vectors.get(face)
                    if now is None or np.isnan(before[face]).any():
                        continue
                    if not np.allclose(now, before[face], atol=1e-9):
                        changed_faces.append([int(origin), face])
        rows.append(
            {
                "program_id": program["program_id"],
                "corrected_steps": steps,
                "faces_whose_reading_changed": changed_faces,
                "reading_changed": bool(changed_faces),
                "newly_stable_origins": newly_stable,
            }
        )
        if (index + 1) % 20 == 0:
            print("  %d/%d  (%.1f min)" % (
                index + 1, len(programs), (time.time() - started) / 60), flush=True)

    np.savez_compressed(
        TENSOR,
        gain=tensor,
        program_ids=np.array([row["program_id"] for row in rows], dtype=object),
        origins=np.array(origins, dtype=np.int64),
        faces=np.array(list(FACES), dtype=object),
        support_a=np.array(support_a, dtype=object),
        support_b=np.array(support_b, dtype=object),
    )
    changed = [row for row in rows if row["reading_changed"]]
    unlocked = [row for row in rows if row["newly_stable_origins"]]
    return {
        "stage": "P4D_PARAM_CORRECTION_RERUN",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_INSTRUMENT_CORRECTION",
        "data_version": gaps.DATA_VERSION,
        "corrects": "artifacts/main_protocol/p4d_gap_repairability_audit.json",
        "supersedes_numbers": False,
        "frozen_p1_common_dsl_edited": False,
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": fits,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "operators_added": 0,
            "thresholds_changed": 0,
        },
        "correction": {
            "parameters": {op: dict(params)
                           for op, params in PARAM_CORRECTIONS.items()},
            "why": {
                "period_complete": (
                    "s1_impute.py:105-109 returns interp_nan(y) when period < 2, "
                    "so at period=0 it is linear interpolation and identical to "
                    "identity; its ZERO_BEHAVIOR reading was the instrument"
                ),
                "impute_ar": (
                    "order resolves to max(8, period); the operator's docstring "
                    "records linear 2.00 / AR(8) 1.33 / AR(24) 0.34, so period=0 "
                    "ran AR(8) and could not see the seasonal lag"
                ),
                "repair_level_shift": (
                    "accepts period but declares no public property for it; "
                    "corrected for consistency with the Classification line"
                ),
            },
            "scope": (
                "declared in this module only; run_forecast_p1._params is the "
                "frozen P1 Common DSL that P1-P4c ran against and is unchanged"
            ),
        },
        "programs_rerun": len(rows),
        "programs_whose_reading_changed": len(changed),
        "programs_newly_stable": [
            {"program_id": row["program_id"],
             "origins": row["newly_stable_origins"]}
            for row in unlocked
        ],
        "rows": rows,
        "corrected_tensor": {
            "path": TENSOR.relative_to(PROJECT_ROOT).as_posix(),
            "shape": list(tensor.shape),
            "axes": ["program", "origin", "face", "series"],
        },
        "wall_seconds": round(time.time() - started, 1),
        "verdict": (
            "CORRECTION_UNLOCKS_NEW_STABLE_PROGRAMS" if unlocked
            else "CORRECTION_CHANGES_READINGS_WITHOUT_NEW_STABILITY" if changed
            else "CORRECTION_IS_INERT"
        ),
        "releases": "NONE",
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("rerun            : %d programs, %d readings changed" % (
        report["programs_rerun"], report["programs_whose_reading_changed"]))
    for entry in report["programs_newly_stable"]:
        print("   NEWLY STABLE  %-36s at %s" % (
            entry["program_id"], entry["origins"]))
    print("consumer fits    : %d in %.1f min" % (
        report["boundary"]["consumer_fits"], report["wall_seconds"] / 60))
    print("verdict          : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
