"""What does a Program actually change, and does it change with the origin?

Two readings the sweep could not give, both mechanical and both 0 Consumer fit.

**Why ``outlier_iqr`` reads +0.0000 on Support-A at every origin.**  The face
names are counter-intuitive: ``ForecastCell.roster("support_a")`` trains on the
Support-B series and evaluates on Support-A (``run_forecast_p1.py:142-146``),
and ``_evaluate`` applies the Program to training windows only.  So a zero gain
on the Support-A face means the Program left the *Support-B* series untouched.
This audit counts the points each operator actually modifies, per face, so a
zero can be attributed to the data rather than filed as an unexplained null.

**Whether six origins are six training conditions.**  Training windows are
``anchor - 192 : anchor + 48`` over ``config["anchors"] = [312 ... 852]``, kept
when ``anchor + HORIZON <= origin``.  Every anchor clears that test at any
origin past 900, and nothing in the P4 path overrides the anchor list, so the
training corpus may be *identical* across all six swept origins -- with only the
evaluation window moving.  If so, "cross-origin" in P4D/P4G varies the scoring
window under a fixed training intervention, which is a weaker statement than it
sounds and matters for how new origins are chosen.

0 LLM calls, 0 Consumer fits, 0 held-out reads.  Nothing is written except this
report.
"""
from __future__ import annotations

import hashlib
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
from evaluation.main_protocol_p4 import audit_param_correction_rerun as fixes
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4h_training_intervention_geometry.json"

CONTEXT, HORIZON = 192, 48
FACES = ("support_a", "support_b")
PROBED = (
    "outlier_iqr", "outlier_mad", "winsorize", "hampel_filter",
    "period_median_complete", "period_complete", "impute_fft", "impute_ar",
)


def _params(op: str) -> dict[str, Any]:
    params = dict(forecast_p1._params(op))
    params.update(fixes.PARAM_CORRECTIONS.get(op, {}))
    return params


def training_windows(values: dict[str, np.ndarray], train_uids: Sequence[str],
                     origin: int) -> list[tuple[str, int, np.ndarray]]:
    """``_evaluate``'s own rule, restated once so the count is comparable."""
    anchors = [int(anchor) for anchor in forecast_p1._config()["anchors"]]
    windows = []
    for uid in train_uids:
        raw = np.asarray(values[uid], dtype=np.float64)
        for anchor in anchors:
            if anchor + HORIZON > origin:
                continue
            windows.append(
                (uid, anchor, raw[anchor - CONTEXT:anchor + HORIZON])
            )
    return windows


def corpus_fingerprint(windows: Sequence[tuple[str, int, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for uid, anchor, window in windows:
        digest.update(uid.encode())
        digest.update(str(anchor).encode())
        digest.update(np.ascontiguousarray(window).tobytes())
    return digest.hexdigest()[:16]


def modified_points(op: str, windows: Sequence[tuple[str, int, np.ndarray]]
                    ) -> dict[str, Any]:
    """Points the operator actually moves, against the linear-integrity baseline."""
    changed = touched_windows = failed = 0
    total = 0
    for _uid, _anchor, window in windows:
        total += int(window.size)
        try:
            baseline = forecast_runtime._linear_integrity(window)
            execution = run_pipeline(
                [(op, _params(op))], window, source="p4h_geometry_audit"
            )
            if not execution.ok or execution.artifact is None:
                failed += 1
                continue
            prepared = forecast_runtime._linear_integrity(
                np.asarray(execution.artifact, dtype=np.float64).ravel()
            )
        except Exception:  # noqa: BLE001 - a refusal is a reading here
            failed += 1
            continue
        moved = int(np.count_nonzero(~np.isclose(prepared, baseline, equal_nan=True)))
        changed += moved
        touched_windows += int(moved > 0)
    return {
        "operator": op,
        "windows": len(windows),
        "windows_touched": touched_windows,
        "points_total": total,
        "points_changed": changed,
        "modified_fraction": round(changed / total, 6) if total else None,
        "windows_failed": failed,
        "is_no_op_on_this_corpus": changed == 0 and failed == 0,
    }


def build() -> dict[str, Any]:
    support_a, support_b, origins = gaps._roster_from_preflight()
    variant = preflight.load_variant()
    values = {uid: variant[uid] for uid in (*support_a, *support_b)}
    # Face -> which series the Program is applied to.
    training_of = {"support_a": support_b, "support_b": support_a}

    per_origin_fingerprint: dict[str, dict[str, str]] = {}
    for face in FACES:
        per_origin_fingerprint[face] = {
            str(origin): corpus_fingerprint(
                training_windows(values, training_of[face], int(origin))
            )
            for origin in origins
        }
    identical = {
        face: len(set(per_origin_fingerprint[face].values())) == 1
        for face in FACES
    }

    # The corpus is the same at every origin when the fingerprints agree, so the
    # modification count only needs to be taken once per face.
    counts = {}
    for face in FACES:
        windows = training_windows(values, training_of[face], int(origins[0]))
        counts[face] = [modified_points(op, windows) for op in PROBED]

    asymmetric = []
    for index, op in enumerate(PROBED):
        a, b = counts["support_a"][index], counts["support_b"][index]
        if a["is_no_op_on_this_corpus"] != b["is_no_op_on_this_corpus"]:
            asymmetric.append(
                {
                    "operator": op,
                    "support_a_face_corpus_is_support_b_series": a,
                    "support_b_face_corpus_is_support_a_series": b,
                }
            )
    return {
        "stage": "P4H_TRAINING_INTERVENTION_GEOMETRY",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_MECHANICAL_AUDIT",
        "data_version": preflight.DATA_VERSION,
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": 0,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "artifacts_modified": 0,
        },
        "face_semantics": (
            "roster('support_a') trains on Support-B and evaluates on Support-A; "
            "_evaluate applies the Program to training rows only, so a gain of "
            "exactly zero on a face means the Program left that face's *training* "
            "series untouched"
        ),
        "anchors": list(forecast_p1._config()["anchors"]),
        "training_corpus_by_origin": {
            "fingerprints": per_origin_fingerprint,
            "identical_across_swept_origins": identical,
            "origins": list(origins),
            "reading": (
                "every anchor clears anchor + 48 <= origin at each swept origin "
                "and no P4 path overrides the anchor list, so the Program's "
                "intervention is the same corpus at all six; only the evaluation "
                "window moves"
                if all(identical.values()) else
                "the training corpus differs across origins"
            ),
        },
        "modified_points_per_face": counts,
        "operators_asymmetric_between_faces": asymmetric,
        "verdict": (
            "ZERO_GAIN_IS_A_NO_OP_ON_THAT_FACES_TRAINING_CORPUS"
            if asymmetric else "NO_FACE_ASYMMETRY_FOUND"
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
    corpus = report["training_corpus_by_origin"]
    print("training corpus identical across origins: %s" % corpus[
        "identical_across_swept_origins"])
    for face in FACES:
        print("--- face %s (Program applied to %s series)" % (
            face, "support_b" if face == "support_a" else "support_a"))
        print("    %-24s %8s %8s %10s" % (
            "operator", "windows", "touched", "modified"))
        for row in report["modified_points_per_face"][face]:
            flag = "   <-- NO-OP" if row["is_no_op_on_this_corpus"] else ""
            print("    %-24s %8d %8d %10d%s" % (
                row["operator"], row["windows"], row["windows_touched"],
                row["points_changed"], flag))
    print("verdict : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
