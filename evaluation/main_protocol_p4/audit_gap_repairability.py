"""P4c's sweep, rerun on the data that actually has gaps -- and kept this time.

P4c exhausted the legal Program space and found nothing stable on both faces.
That sweep ran on ``series_cache.npz``, which was built from the *without*-missing
KDD release, so every imputation operator was probed against a series with
nothing to impute.  ``AGENTS.md`` §8.1 records the erratum; this audit reruns the
identical program space on ``EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING``.

Two deliberate differences from P4c, and no others:

* the data carries its natural 17.1% gaps, and the roster is recomputed on it
  (239/270 readable, so the A/B membership is *not* P4c's);
* the full ``program x origin x face x series`` gain tensor is persisted to an
  ``.npz`` beside the report.  P4c computed those vectors and dropped them at
  write time (``audit_program_repairability.py:252-259``), which is why the
  cross-fitted targeting question needed a rerun to ask at all.

What identity means here matters for reading any null result.
``_apply_program(window, None)`` is ``_linear_integrity(window)``, and every
program's output passes through ``_linear_integrity`` as well, so the baseline
*is* linear interpolation.  ``impute_linear`` is therefore ZERO_BEHAVIOR by
construction; the live question is whether fft / ema / period-median / ar
completion beat linear interpolation.

Origins whose horizon leaves some eval series with no observed truth are dropped
rather than repaired: filling them would score predictions against interpolation.
No LLM, no Outcome, no held-out origin, no new operator, no threshold change.
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
from evaluation.main_protocol_p4 import audit_program_repairability as p4c
from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT = PROJECT_ROOT / "artifacts/main_protocol/p4d_gap_repairability_audit.json"
TENSOR = PROJECT_ROOT / "artifacts/main_protocol/p4d_gap_per_series_gain.npz"
PREFLIGHT = PROJECT_ROOT / "artifacts/main_protocol/p4d_natural_gap_preflight.json"

FACES = p4c.FACES
MATERIAL = p4c.MATERIAL
DATA_VERSION = preflight.DATA_VERSION


def _roster_from_preflight() -> tuple[tuple[str, ...], tuple[str, ...], list[int]]:
    """The gapped-data roster and usable origins, as the preflight measured them."""
    if not PREFLIGHT.is_file():
        raise RuntimeError("run preflight_natural_gap_variant first")
    payload = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if payload.get("verdict") != "PREFLIGHT_PASS":
        raise RuntimeError("preflight did not pass: %s" % payload.get("verdict"))
    geometry = payload["gates"]["gate_2_roster_geometry"]
    truth = payload["gates"]["gate_3_horizon_truth"]
    return (
        tuple(geometry["support_a"]),
        tuple(geometry["support_b"]),
        [int(origin) for origin in truth["usable_origins"]],
    )


def _variant_cell(support_a: tuple[str, ...],
                  support_b: tuple[str, ...]) -> forecast_p1.ForecastCell:
    variant = preflight.load_variant()
    values = {uid: variant[uid] for uid in (*support_a, *support_b)}
    return forecast_p1.ForecastCell(
        values=values,
        support_a=support_a,
        support_b=support_b,
        observation_block=np.asarray(
            values[support_a[0]][:forecast_p1.ORIGIN], dtype=np.float64
        ),
    )


def sweep(cell: forecast_p1.ForecastCell, origins: Sequence[int],
          programs: Sequence[Mapping[str, Any]], *,
          progress_every: int = 25) -> dict[str, Any]:
    """Every program on every face at every origin, keeping the per-series gains."""
    started = time.time()
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
    fits = len(origins) * len(FACES)

    series_count = len(cell.support_a)
    tensor = np.full(
        (len(programs), len(origins), len(FACES), series_count), np.nan,
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    for index, program in enumerate(programs):
        per_origin = []
        for o_index, origin in enumerate(origins):
            faces = {}
            for f_index, face in enumerate(FACES):
                reading = p4c._face_reading(
                    executors[(int(origin), face)], program["steps"], int(origin)
                )
                fits += 1
                faces[face] = reading
                gains = reading.get("per_series_gain")
                if gains is not None and len(gains) == series_count:
                    tensor[index, o_index, f_index, :] = gains
            verdicts = {
                face: p4c._verdict(faces[face], contract.BOUNDED_POLICY)
                for face in FACES
            }
            per_origin.append(
                {
                    "origin": int(origin),
                    "stable_on_both_faces": all(
                        verdicts[face].get("admitted") for face in FACES
                    ),
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
        unreadable = [
            entry["origin"] for entry in per_origin
            if any("failed" in entry[face] for face in FACES)
        ]
        rows.append(
            {
                "program_index": index,
                "program_id": program["program_id"],
                "steps": program["steps"],
                "workflow_length": program["workflow_length"],
                "targeting_modes": program["targeting_modes"],
                "stable_origins": [
                    entry["origin"] for entry in per_origin
                    if entry["stable_on_both_faces"]
                ],
                "a_face_admitted_origins": [
                    entry["origin"] for entry in per_origin
                    if entry["verdicts"]["support_a"]["admitted"]
                ],
                "b_face_admitted_origins": [
                    entry["origin"] for entry in per_origin
                    if entry["verdicts"]["support_b"]["admitted"]
                ],
                "unreadable_origins": unreadable,
                "readable_origin_count": len(per_origin) - len(unreadable),
                "per_origin": per_origin,
            }
        )
        if (index + 1) % progress_every == 0:
            print("  %d/%d programs  (%.1f min)" % (
                index + 1, len(programs), (time.time() - started) / 60), flush=True)
    return {
        "rows": rows,
        "tensor": tensor,
        "identity": identity,
        "consumer_fits": fits,
        "wall_seconds": round(time.time() - started, 1),
    }


def _imputation_reading(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Did any completion operator beat linear interpolation, and where?

    Reported separately from the whole space because it is the one question the
    without-missing line could not ask at all.
    """
    families = ("impute_fft", "impute_ema", "impute_ar", "period_median_complete",
                "period_complete", "impute_linear")
    out = []
    for row in rows:
        ops = [str(step["op"]) for step in row["steps"]]
        if not any(op in families for op in ops):
            continue
        best_a = max(
            (entry["support_a"].get("aggregate_gain")
             for entry in row["per_origin"]
             if entry["support_a"].get("aggregate_gain") is not None),
            default=None,
        )
        out.append(
            {
                "program_id": row["program_id"],
                "single_step": row["workflow_length"] == 1,
                "stable_origins": row["stable_origins"],
                "a_face_admitted_origins": row["a_face_admitted_origins"],
                "b_face_admitted_origins": row["b_face_admitted_origins"],
                "best_support_a_gain": best_a,
            }
        )
    out.sort(key=lambda entry: (-len(entry["stable_origins"]),
                                -(entry["best_support_a_gain"] or -9e9)))
    return {
        "programs_touching_completion": len(out),
        "stable_on_both_faces_anywhere": [
            entry for entry in out if entry["stable_origins"]
        ],
        "top_by_support_a_gain": out[:15],
    }


def build(*, two_step: bool, progress_every: int) -> dict[str, Any]:
    support_a, support_b, origins = _roster_from_preflight()
    cell = _variant_cell(support_a, support_b)
    programs = p4c.program_space(two_step=two_step)
    print("programs %d | origins %s | faces %d" % (
        len(programs), origins, len(FACES)), flush=True)
    result = sweep(cell, origins, programs, progress_every=progress_every)
    rows = result["rows"]

    np.savez_compressed(
        TENSOR,
        gain=result["tensor"],
        program_ids=np.array([row["program_id"] for row in rows], dtype=object),
        origins=np.array(origins, dtype=np.int64),
        faces=np.array(list(FACES), dtype=object),
        support_a=np.array(support_a, dtype=object),
        support_b=np.array(support_b, dtype=object),
        data_version=np.array([DATA_VERSION], dtype=object),
    )

    legal = [row for row in rows if row["readable_origin_count"] > 0]
    stable = [row for row in legal if row["stable_origins"]]
    return {
        "stage": "P4D_GAP_REPAIRABILITY_AUDIT",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_EXHAUSTIVE_PROGRAM_SWEEP",
        "data_version": DATA_VERSION,
        "does_not_merge_with": preflight.INCUMBENT_VERSION,
        "parallel_to": "artifacts/main_protocol/p4c_program_repairability_audit.json",
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": result["consumer_fits"],
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "natural_final_outcome_reads": 0,
            "thresholds_changed": 0,
            "operators_added": 0,
        },
        "geometry": {
            "origins_swept": origins,
            "origins_dropped": [1416, 1656],
            "why_dropped": (
                "an eval series has no observed truth in that horizon; scoring "
                "it would compare predictions against interpolation"
            ),
            "support_a": list(support_a),
            "support_b": list(support_b),
            "series_per_face": len(support_a),
        },
        "identity_semantics": (
            "identity == _linear_integrity, so impute_linear is ZERO_BEHAVIOR by "
            "construction and the live question is whether fft / ema / "
            "period-median / ar completion beat linear interpolation"
        ),
        "space": {
            "programs_enumerated": len(rows),
            "programs_readable_somewhere": len(legal),
            "programs_stable_on_both_faces_somewhere": len(stable),
        },
        "completion_family": _imputation_reading(rows),
        "stable_programs": [
            {
                "program_id": row["program_id"],
                "steps": row["steps"],
                "stable_origins": row["stable_origins"],
            }
            for row in stable
        ],
        "per_series_gain_tensor": {
            "path": TENSOR.relative_to(PROJECT_ROOT).as_posix(),
            "shape": list(result["tensor"].shape),
            "axes": ["program", "origin", "face", "series"],
            "note": (
                "NaN where the program was refused or unreadable on that face; "
                "this is the matrix P4c computed and discarded"
            ),
        },
        "wall_seconds": result["wall_seconds"],
        "verdict": (
            "GAP_REPAIR_STABLE_CANDIDATE_FOUND" if stable
            else "NO_STABLE_CANDIDATE_ON_NATURAL_GAPS"
        ),
        "releases": "NONE",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-step-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)

    report = build(two_step=not args.single_step_only,
                   progress_every=args.progress_every)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    space = report["space"]
    print("programs          : %d enumerated, %d readable, %d stable on both faces"
          % (space["programs_enumerated"], space["programs_readable_somewhere"],
             space["programs_stable_on_both_faces_somewhere"]))
    completion = report["completion_family"]
    print("completion family : %d programs, %d stable somewhere" % (
        completion["programs_touching_completion"],
        len(completion["stable_on_both_faces_anywhere"])))
    for entry in completion["top_by_support_a_gain"][:8]:
        print("   %-34s A-admit %s  B-admit %s  best A gain %s" % (
            entry["program_id"], entry["a_face_admitted_origins"],
            entry["b_face_admitted_origins"],
            None if entry["best_support_a_gain"] is None
            else round(entry["best_support_a_gain"], 4)))
    print("consumer fits     : %d in %.1f min" % (
        report["boundary"]["consumer_fits"], report["wall_seconds"] / 60))
    print("verdict           : %s" % report["verdict"])
    print("wrote %s" % REPORT.relative_to(PROJECT_ROOT).as_posix())
    print("wrote %s" % TENSOR.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
