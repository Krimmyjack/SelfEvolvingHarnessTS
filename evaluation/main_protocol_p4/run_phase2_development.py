"""Phase 2, cohort 1: collect the O0 x view gain tensor the menu is chosen from.

This is the development half.  It reads cohort 1 -- already-read data -- and
produces the tensor that the menu, the Best-Fixed choice and the Targeter are
all derived from.  Nothing here touches cohort 2; the confirmation run is a
separate module so that no code path can select on the confirmation Outcome.

Design points worth stating once:

* the program space is the frozen O0 enumeration **deduplicated by effect** on
  cohort 1, plus the empty program.  Aliases arise because four of the eighteen
  operators are no-ops on gapped data, and an alias would spend menu budget
  twice on one discovery.  Deduplication survives the view: a view is applied
  after the program, so two programs with identical prepared arrays stay
  identical under any view.
* every cell is scored against **one** reference -- the empty program under the
  identity view -- so the four view rows are commensurable and "utility vs raw"
  means the same thing in all of them.
* the identity view row is recomputed rather than reused from P4D, so the whole
  Phase-2 matrix is internally consistent under the corrected parameters.

0 LLM calls, 0 held-out reads, 0 UCR TEST bytes.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p4 import audit_cross_fitted_targeting as targeting
from evaluation.main_protocol_p4 import audit_gap_repairability as gaps
from evaluation.main_protocol_p4 import audit_param_correction_rerun as fixes
from evaluation.main_protocol_p4 import audit_program_repairability as p4c
from evaluation.main_protocol_p4 import phase2_contract as contract
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT = PROJECT_ROOT / "artifacts/main_protocol/p4k_phase2_development.json"
TENSOR = PROJECT_ROOT / "artifacts/main_protocol/p4k_phase2_development_gain.npz"

FACES = contract.FACES
VIEW_BY_NAME = {view.name: view for view in views.CANDIDATE_VIEWS}
EMPTY = "raw"


def frozen_program_space() -> list[dict[str, Any]]:
    """The empty program, then cohort 1's distinct readable effects."""
    data = targeting.load_tensor()
    gain, ids, origins = data["gain"], data["program_ids"], data["origins"]
    columns = [origins.index(origin) for origin in contract.ORIGINS]
    seen: dict[bytes, str] = {}
    for index in range(gain.shape[0]):
        block = gain[index][columns]
        if bool(np.isnan(block).any()):
            continue
        seen.setdefault(np.round(block, 9).tobytes(), ids[index])
    distinct = sorted(seen.values())

    by_id = {
        program["program_id"]: program
        for program in p4c.program_space(two_step=True)
    }
    programs: list[dict[str, Any]] = [
        {"program_id": EMPTY, "steps": [], "workflow_length": 0}
    ]
    for program_id in distinct:
        source = by_id[program_id]
        programs.append({
            "program_id": program_id,
            "steps": fixes.corrected_steps(source["steps"]),
            "workflow_length": source["workflow_length"],
        })
    return programs


def sweep(cell: Any, programs: Sequence[Mapping[str, Any]],
          *, progress_every: int = 20) -> dict[str, Any]:
    started = time.time()
    origins = list(contract.ORIGINS)
    view_names = list(contract.O1_VIEWS)
    series = len(cell.support_a)

    smase = np.full(
        (len(programs), len(view_names), len(origins), len(FACES), series),
        np.nan, dtype=np.float64,
    )
    reference = np.full((len(origins), len(FACES), series), np.nan, np.float64)
    fits = 0
    refused: list[dict[str, Any]] = []

    executors: dict[tuple[int, str], Any] = {}
    rosters: dict[tuple[int, str], Any] = {}
    configs: dict[int, Any] = {}
    for origin in origins:
        at = forecast_p4._cell_at(cell, int(origin))
        configs[int(origin)] = forecast_p4._config(int(origin))
        for face in FACES:
            rosters[(int(origin), face)] = at.roster(face)
            executors[(int(origin), face)] = at

    for p_index, program in enumerate(programs):
        compiled_cache: dict[tuple[int, str], Any] = {}
        for v_index, view_name in enumerate(view_names):
            view = VIEW_BY_NAME[view_name]
            for o_index, origin in enumerate(origins):
                at = executors[(int(origin), FACES[0])]
                for f_index, face in enumerate(FACES):
                    roster = rosters[(int(origin), face)]
                    config = configs[int(origin)]
                    steps = tuple(
                        (str(s["op"]), dict(s["params"]))
                        for s in program["steps"]
                    )
                    key = (int(origin), face)
                    if steps and key not in compiled_cache:
                        executor = p4c.ScopeExecutor(
                            roster, at.values, config,
                            evaluate_fn=views.forecast_runtime._evaluate,
                            max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION,
                        )
                        verification = executor.verify(steps, int(origin))
                        compiled_cache[key] = (
                            executor._compiled(steps)
                            if verification.passed else None
                        )
                    compiled = compiled_cache.get(key) if steps else None
                    if steps and compiled is None:
                        refused.append({
                            "program_id": program["program_id"],
                            "view": view_name, "origin": int(origin),
                            "face": face, "reason": "WINDOW_VERIFIER_REJECTED",
                        })
                        continue
                    try:
                        reading = views.representation_evaluate(
                            roster, at.values, compiled, config,
                            origin=int(origin), view=view,
                        )
                    except Exception as exc:  # noqa: BLE001 - refusal is a reading
                        refused.append({
                            "program_id": program["program_id"],
                            "view": view_name, "origin": int(origin),
                            "face": face,
                            "reason": "%s: %s" % (type(exc).__name__, str(exc)[:90]),
                        })
                        continue
                    fits += 1
                    losses = np.asarray(
                        reading["per_view_smase"], dtype=np.float64)
                    smase[p_index, v_index, o_index, f_index, :] = losses
                    if program["program_id"] == EMPTY and view_name == "identity":
                        reference[o_index, f_index, :] = losses
        if (p_index + 1) % progress_every == 0:
            print("  %d/%d programs  (%.1f min, %d fits)" % (
                p_index + 1, len(programs), (time.time() - started) / 60, fits),
                flush=True)

    if bool(np.isnan(reference).any()):
        raise RuntimeError("the raw reference did not evaluate everywhere")
    gain = reference[None, None, :, :, :] - smase
    return {
        "smase": smase, "gain": gain, "reference": reference,
        "consumer_fits": fits, "refused": refused,
        "wall_seconds": round(time.time() - started, 1),
    }


def _headroom(gain: np.ndarray, view_names: Sequence[str]) -> dict[str, Any]:
    """Best achievable mean gain per view -- an upper bound, never a result."""
    rows = []
    for v_index, name in enumerate(view_names):
        block = gain[:, v_index, :, :, :]
        per_cell = np.nanmean(block, axis=-1)  # program x origin x face
        with np.errstate(invalid="ignore"):
            best_fixed = np.nanmax(np.nanmean(per_cell, axis=(1, 2)))
            best_per_cell = np.nanmax(per_cell, axis=0)
        rows.append({
            "view": name,
            "best_single_program_mean_gain": round(float(best_fixed), 6),
            "best_per_cell_mean_gain": round(float(np.mean(best_per_cell)), 6),
        })
    return {
        "per_view": rows,
        "note": contract.ORACLE_IS_NOT_A_RESULT,
    }


def build() -> dict[str, Any]:
    frozen = contract.assert_frozen()
    if not frozen["frozen"]:
        raise RuntimeError("Phase-2 contract drifted: %s" % frozen["failures"])
    support_a, support_b, _origins = gaps._roster_from_preflight()
    cell = gaps._variant_cell(support_a, support_b)
    programs = frozen_program_space()
    print("cohort 1 | programs %d (incl. raw) | views %d | origins %s" % (
        len(programs), len(contract.O1_VIEWS), list(contract.ORIGINS)), flush=True)

    result = sweep(cell, programs)
    np.savez_compressed(
        TENSOR,
        gain=result["gain"],
        smase=result["smase"],
        reference=result["reference"],
        program_ids=np.array([p["program_id"] for p in programs], dtype=object),
        views=np.array(list(contract.O1_VIEWS), dtype=object),
        origins=np.array(list(contract.ORIGINS), dtype=np.int64),
        faces=np.array(list(FACES), dtype=object),
        support_a=np.array(support_a, dtype=object),
        support_b=np.array(support_b, dtype=object),
    )
    return {
        "stage": "P4K_PHASE2_DEVELOPMENT",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_MENU_CONSTRUCTION",
        "data_version": contract.DATA_VERSION,
        "cohort": "cohort_1",
        "cohort_role": contract.COHORTS["cohort_1"]["role"],
        "frozen_contract": frozen,
        "gain_reference": contract.GAIN_REFERENCE,
        "boundary": {
            **contract.BOUNDARY,
            "consumer_fits": result["consumer_fits"],
        },
        "space": {
            "programs_including_raw": len(programs),
            "deduplication": contract.O0_DEDUPLICATION,
            "views": list(contract.O1_VIEWS),
        },
        "refusals": {
            "count": len(result["refused"]),
            "rows": result["refused"][:40],
        },
        "headroom_upper_bound": _headroom(
            result["gain"], list(contract.O1_VIEWS)),
        "tensor": {
            "path": TENSOR.relative_to(PROJECT_ROOT).as_posix(),
            "shape": list(result["gain"].shape),
            "axes": ["program", "view", "origin", "face", "series"],
        },
        "wall_seconds": result["wall_seconds"],
        "releases": (
            "the cohort-1 tensor only; the menu, Best-Fixed and Targeter are "
            "frozen by the confirmation runner before cohort 2 is read"
        ),
    }


def main() -> int:
    report = build()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("programs      : %d" % report["space"]["programs_including_raw"])
    print("refusals      : %d" % report["refusals"]["count"])
    print("%-28s %14s %14s" % ("view", "best fixed", "best per cell"))
    for row in report["headroom_upper_bound"]["per_view"]:
        print("%-28s %+14.6f %+14.6f" % (
            row["view"], row["best_single_program_mean_gain"],
            row["best_per_cell_mean_gain"]))
    print("consumer fits : %d in %.1f min" % (
        report["boundary"]["consumer_fits"], report["wall_seconds"] / 60))
    print("wrote %s" % REPORT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
