"""Where does the origin-2856 gain actually come from?

The exhaustive sweep found three programs stable on both faces at origin 2856,
and all three begin with ``period_median_complete``.  That is suggestive, not an
attribution: the sweep never ran completion alone, never ran the outlier step
alone, and never ran the two in the other order.  Three things are therefore
still unseparated -- the gap, the composition, and the order.

This is the minimal design that separates them.  One roster, one origin set,
five arm families, two data versions:

* **gap** -- every arm runs on the with-missing variant *and* on the filled
  cache, same UIDs and same origins.  A gain that survives the fill was never
  about gaps.
* **composition** -- completion alone and outlier alone run beside the pair, so
  the pair can be compared against the better of its parts rather than against
  identity.
* **order** -- ``completion > outlier`` and ``outlier > completion`` both run.
  At origin 2136 the sweep already found ``winsorize>outlier_mad`` and
  ``outlier_mad>winsorize`` numerically identical, so commutativity is a real
  possibility that has to be measured rather than assumed.

``period_complete`` is included with the corrected ``period`` (see
``audit_param_correction_rerun``); on the filled cache both completion operators
must be inert by construction, which is the design's own sanity check.

0 LLM calls, no threshold change, no operator added, no held-out read.
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
from evaluation.main_protocol_p4 import audit_param_correction_rerun as fixes
from evaluation.main_protocol_p4 import audit_program_repairability as p4c
from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4f_matched_composition.json"
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"

FACES = gaps.FACES
MATERIAL = gaps.MATERIAL

COMPLETION = ("period_median_complete", "period_complete")
OUTLIER = ("outlier_iqr", "outlier_mad", "winsorize")


def _params(op: str) -> dict[str, Any]:
    params = dict(forecast_p1._params(op))
    params.update(fixes.PARAM_CORRECTIONS.get(op, {}))
    return params


def arms() -> list[dict[str, Any]]:
    """Identity, each part alone, and both orders of every pair."""
    rows: list[dict[str, Any]] = [
        {"arm_id": "identity", "family": "baseline", "steps": []}
    ]
    for op in COMPLETION:
        rows.append({"arm_id": op, "family": "completion_alone",
                     "steps": [{"op": op, "params": _params(op)}]})
    for op in OUTLIER:
        rows.append({"arm_id": op, "family": "outlier_alone",
                     "steps": [{"op": op, "params": _params(op)}]})
    for completion in COMPLETION:
        for outlier in OUTLIER:
            rows.append({
                "arm_id": "%s>%s" % (completion, outlier),
                "family": "completion_then_outlier",
                "steps": [{"op": completion, "params": _params(completion)},
                          {"op": outlier, "params": _params(outlier)}],
            })
            rows.append({
                "arm_id": "%s>%s" % (outlier, completion),
                "family": "outlier_then_completion",
                "steps": [{"op": outlier, "params": _params(outlier)},
                          {"op": completion, "params": _params(completion)}],
            })
    return rows


def _cache_values(uids: Sequence[str]) -> dict[str, np.ndarray]:
    """The same UIDs read from the upstream-filled cache."""
    cache = np.load(CACHE, allow_pickle=True)
    names = [str(value) for value in cache["names"]]
    rows = cache["values"]
    position = {name: index for index, name in enumerate(names)}
    return {
        uid: np.asarray(rows[position[uid]], dtype=np.float64) for uid in uids
    }


def _cell(values: Mapping[str, np.ndarray], support_a: tuple[str, ...],
          support_b: tuple[str, ...]) -> forecast_p1.ForecastCell:
    return forecast_p1.ForecastCell(
        values=dict(values), support_a=support_a, support_b=support_b,
        observation_block=np.asarray(
            values[support_a[0]][:forecast_p1.ORIGIN], dtype=np.float64
        ),
    )


def _admits(vector: np.ndarray | None) -> bool | None:
    if vector is None or vector.size == 0 or bool(np.isnan(vector).any()):
        return None
    worst = float(-vector.min())
    return bool(
        float(vector.mean()) >= MATERIAL
        and float((vector < -MATERIAL).mean()) <= contract.BOUNDED_MAX_HARMED_FRACTION
        and max(worst, 0.0) <= contract.BOUNDED_MAX_SINGLE_SERIES_HARM
    )


def run_version(label: str, values: Mapping[str, np.ndarray],
                support_a: tuple[str, ...], support_b: tuple[str, ...],
                origins: Sequence[int], plan: Sequence[Mapping[str, Any]],
                ) -> tuple[dict[str, Any], np.ndarray, int]:
    cell = _cell(values, support_a, support_b)
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
    tensor = np.full(
        (len(plan), len(origins), len(FACES), len(support_a)), np.nan, dtype=np.float64
    )
    fits = len(origins) * len(FACES)
    rows = []
    for index, arm in enumerate(plan):
        per_origin = []
        for o_index, origin in enumerate(origins):
            entry: dict[str, Any] = {"origin": int(origin)}
            vectors: dict[str, np.ndarray | None] = {}
            for f_index, face in enumerate(FACES):
                if not arm["steps"]:
                    gains = [0.0] * len(support_a)
                else:
                    reading = p4c._face_reading(
                        executors[(int(origin), face)], arm["steps"], int(origin)
                    )
                    fits += 1
                    gains = reading.get("per_series_gain")
                    if gains is None:
                        entry[face] = {"failed": reading.get("failed")}
                        vectors[face] = None
                        continue
                vector = np.asarray(gains, dtype=np.float64)
                tensor[index, o_index, f_index, :] = vector
                vectors[face] = vector
                entry[face] = {
                    "aggregate_gain": round(float(vector.mean()), 6),
                    "harmed_series": int((vector < -MATERIAL).sum()),
                    "max_single_series_harm": round(
                        max(0.0, float(-vector.min())), 6),
                    "admitted": _admits(vector),
                }
            entry["stable_on_both_faces"] = all(
                _admits(vectors.get(face)) for face in FACES
            )
            per_origin.append(entry)
        rows.append({
            "arm_id": arm["arm_id"],
            "family": arm["family"],
            "stable_origins": [
                entry["origin"] for entry in per_origin
                if entry["stable_on_both_faces"]
            ],
            "per_origin": per_origin,
        })
    return {"data_version": label, "arms": rows}, tensor, fits


def _attribution(with_missing: Mapping[str, Any],
                 without: Mapping[str, Any]) -> dict[str, Any]:
    """Gap, composition and order, each read off the matched arms."""
    def by_id(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {row["arm_id"]: row for row in payload["arms"]}

    gapped, filled = by_id(with_missing), by_id(without)

    def best_gain(row: Mapping[str, Any], origin: int, face: str) -> float | None:
        for entry in row["per_origin"]:
            if entry["origin"] == origin:
                return (entry.get(face) or {}).get("aggregate_gain")
        return None

    origins = [entry["origin"] for entry in next(iter(gapped.values()))["per_origin"]]
    composition = []
    for completion in COMPLETION:
        for outlier in OUTLIER:
            forward, reverse = "%s>%s" % (completion, outlier), "%s>%s" % (outlier, completion)
            for origin in origins:
                row = {"origin": origin, "completion": completion, "outlier": outlier}
                for face in FACES:
                    pair = best_gain(gapped[forward], origin, face)
                    parts = [
                        best_gain(gapped[completion], origin, face),
                        best_gain(gapped[outlier], origin, face),
                    ]
                    parts = [value for value in parts if value is not None]
                    row[face] = {
                        "pair": pair,
                        "best_part": max(parts) if parts else None,
                        "excess_over_best_part": (
                            None if pair is None or not parts
                            else round(pair - max(parts), 6)
                        ),
                        "reverse": best_gain(gapped[reverse], origin, face),
                        "order_effect": (
                            None if pair is None
                            or best_gain(gapped[reverse], origin, face) is None
                            else round(
                                pair - best_gain(gapped[reverse], origin, face), 6)
                        ),
                        "same_arm_on_filled_data": best_gain(
                            filled[forward], origin, face),
                    }
                composition.append(row)
    completion_inert_on_filled = all(
        (best_gain(filled[op], origin, face) or 0.0) == 0.0
        for op in COMPLETION for origin in origins for face in FACES
    )
    return {
        "completion_is_inert_on_filled_cache": completion_inert_on_filled,
        "sanity_check_reading": (
            "completion operators do nothing once the gaps are filled, as they "
            "must" if completion_inert_on_filled else
            "a completion operator moved the filled cache; the arm is not "
            "measuring gap repair"
        ),
        "per_pair_per_origin": composition,
    }


def build() -> dict[str, Any]:
    started = time.time()
    support_a, support_b, origins = gaps._roster_from_preflight()
    plan = arms()
    variant = preflight.load_variant()
    gapped_values = {uid: variant[uid] for uid in (*support_a, *support_b)}
    filled_values = _cache_values((*support_a, *support_b))

    print("arms %d | origins %s | two data versions" % (len(plan), origins), flush=True)
    with_missing, gap_tensor, fits_a = run_version(
        preflight.DATA_VERSION, gapped_values, support_a, support_b, origins, plan)
    print("  with-missing done", flush=True)
    without, fill_tensor, fits_b = run_version(
        preflight.INCUMBENT_VERSION, filled_values, support_a, support_b, origins, plan)
    print("  without-missing done", flush=True)

    np.savez_compressed(
        PROJECT_ROOT / "artifacts/main_protocol/p4f_matched_gain.npz",
        with_missing=gap_tensor, without_missing=fill_tensor,
        arm_ids=np.array([arm["arm_id"] for arm in plan], dtype=object),
        origins=np.array(origins, dtype=np.int64),
        faces=np.array(list(FACES), dtype=object),
        support_a=np.array(support_a, dtype=object),
        support_b=np.array(support_b, dtype=object),
    )
    return {
        "stage": "P4F_MATCHED_COMPOSITION_CONTRAST",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_MATCHED_ARM_CONTRAST",
        "question": (
            "is the origin-2856 gain attributable to the gaps, to the "
            "composition, or to the order"
        ),
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": fits_a + fits_b,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "operators_added": 0,
            "thresholds_changed": 0,
        },
        "design": {
            "roster_support_a": list(support_a),
            "roster_support_b": list(support_b),
            "origins": list(origins),
            "arms": [{"arm_id": arm["arm_id"], "family": arm["family"],
                      "steps": arm["steps"]} for arm in plan],
            "data_versions": [preflight.DATA_VERSION, preflight.INCUMBENT_VERSION],
            "matched_on": "same UIDs, same origins, same faces, same programs",
        },
        "with_missing": with_missing,
        "without_missing": without,
        "attribution": _attribution(with_missing, without),
        "wall_seconds": round(time.time() - started, 1),
        "releases": "NONE",
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for payload in (report["with_missing"], report["without_missing"]):
        print("--- %s" % payload["data_version"])
        for row in payload["arms"]:
            if row["arm_id"] == "identity":
                continue
            print("   %-46s stable at %s" % (row["arm_id"], row["stable_origins"]))
    attribution = report["attribution"]
    print("completion inert on filled cache: %s"
          % attribution["completion_is_inert_on_filled_cache"])
    print("consumer fits : %d in %.1f min" % (
        report["boundary"]["consumer_fits"], report["wall_seconds"] / 60))
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
