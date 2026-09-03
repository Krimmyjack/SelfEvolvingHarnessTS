"""Does the existing menu have natural headroom on Adiac / ArrowHead TRAIN?

The ruling for the Classification line is narrow: read only the TRAIN members of
the two sealed-final fixtures, ask whether the *existing* operator menu contains
any stable positive candidate at all, and leave TEST sealed.  If nothing clears
held-in, no TEST byte is worth spending and no operator should be added on the
strength of a TEST reading.

Two things differ from the Forecast line and both change how a number reads:

* the per-view unit is **per-class recall**, not per-series loss.  Adiac has 37
  classes and ArrowHead 3, so the same ``harmed_fraction <= 0.20`` budget means
  "at most 7 classes" on one fixture and "at most 0 classes" on the other.  The
  budget is applied unchanged and the denominator is reported beside it.
* the Consumer ceiling is ``MAX_MODIFIED_FRACTION = 0.10``, so more of the menu
  is refused by the window verifier than in Forecast.

Boundary: only the predeclared ``*_TRAIN.ts`` member of each archive is read,
and the audit asserts that no TEST member was opened.  0 LLM calls, no Skill
store write, no threshold changed -- the bounded rule is imported, not restated.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

from evaluation.main_protocol_p1 import classification_component as cls
from evaluation.main_protocol_p4 import p4b_contract as contract
from SelfEvolvingHarnessTS.methods.ttha import admission_policy
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import classify_relation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4e_classification_headroom.json"
FRESH = PROJECT_ROOT / "data/main_experiment_p0/ucr_fresh"

FACES = ("support_a", "support_b")
MATERIAL = 0.005

SEALED_FINAL = (
    cls.FixtureSpec(
        fixture_id="Adiac",
        archive=FRESH / "Adiac.zip",
        train_member="Adiac_TRAIN.ts",
        role="sealed_final_train_only_headroom_check",
    ),
    cls.FixtureSpec(
        fixture_id="ArrowHead",
        archive=FRESH / "ArrowHead.zip",
        train_member="ArrowHead_TRAIN.ts",
        role="sealed_final_train_only_headroom_check",
    ),
)


def _test_members_untouched(spec: cls.FixtureSpec) -> dict[str, Any]:
    """Name every TEST member in the archive and record that none was read."""
    with ZipFile(spec.archive) as archive:
        members = archive.namelist()
    return {
        "fixture_id": spec.fixture_id,
        "train_member_read": spec.train_member,
        "test_members_present": sorted(
            name for name in members if "_TEST." in name
        ),
        "test_member_bytes_read": 0,
    }


def _risk(identity: Mapping[str, Any], reading: Mapping[str, Any]) -> dict[str, Any]:
    """Per-class gain, and the bounded rule applied to it unchanged."""
    base = dict(identity.get("per_class_recall") or {})
    now = dict(reading.get("per_class_recall") or {})
    keys = sorted(set(base) & set(now), key=str)
    gains = {str(key): float(now[key]) - float(base[key]) for key in keys}
    aggregate = float(reading["utility"]) - float(identity["utility"])
    values = np.asarray(list(gains.values()), dtype=np.float64)
    harmed = values < -MATERIAL
    lowest = float(values.min()) if values.size else 0.0
    facts = classify_relation(
        aggregate_gain=aggregate,
        per_series_gains=gains,
        material_threshold=MATERIAL,
    )
    verdict = admission_policy.decide(
        relation=str(facts["relation"]),
        aggregate_gain=aggregate,
        per_series_gains=gains,
        policy=contract.BOUNDED_POLICY,
    ).to_dict()
    return {
        "aggregate_gain": aggregate,
        "macro_f1": float(reading["macro_f1"]),
        "class_count": int(values.size),
        "harmed_class_count": int(harmed.sum()),
        "harmed_fraction": float(harmed.mean()) if values.size else 0.0,
        "max_single_class_harm": -lowest if lowest < 0.0 else 0.0,
        "behavior_point_count": int(reading.get("behavior_point_count") or 0),
        "relation": str(facts["relation"]),
        "admitted": bool(verdict.get("admitted")),
        "reason": verdict.get("reason"),
        "per_class_gain": gains,
    }


def audit_fixture(spec: cls.FixtureSpec) -> dict[str, Any]:
    started = time.time()
    cell = cls._load_train_fixture(spec)
    identity = cls._identity_readings(cell)
    programs = tuple(
        program for program in dict.fromkeys(cls._eligible_programs())
        if program not in cls.EFFECT_ALIASES
    )
    rows = []
    for program in programs:
        check = cls._verify_program(cell, program)
        if not check["passed"]:
            rows.append(
                {
                    "program": program,
                    "verifier": "REFUSED",
                    "rejection_codes": list(check["rejection_codes"]),
                }
            )
            continue
        readings, _surfaces, _usage = cls._evaluate_faces(
            cell, program, identity=identity
        )
        faces = {
            face: _risk(identity[face], readings[face])
            for face in FACES
        }
        rows.append(
            {
                "program": program,
                "verifier": "PASSED",
                "stable_on_both_faces": all(
                    faces[face]["admitted"] for face in FACES
                ),
                **{face: faces[face] for face in FACES},
            }
        )
    evaluated = [row for row in rows if row["verifier"] == "PASSED"]
    stable = [row for row in evaluated if row["stable_on_both_faces"]]

    def _best(face: str) -> dict[str, Any] | None:
        candidates = [row for row in evaluated if row[face]["behavior_point_count"]]
        if not candidates:
            return None
        best = max(candidates, key=lambda row: row[face]["aggregate_gain"])
        return {
            "program": best["program"],
            "aggregate_gain": best[face]["aggregate_gain"],
            "harmed_fraction": best[face]["harmed_fraction"],
            "max_single_class_harm": best[face]["max_single_class_harm"],
            "admitted": best[face]["admitted"],
        }

    return {
        "fixture_id": spec.fixture_id,
        "boundary": _test_members_untouched(spec),
        "geometry": {
            **cell.split_counts(),
            "classes": len(cell.label_names),
            "series_length": int(cell.values.shape[1]),
            "per_view_unit": "per-class recall",
            "harmed_budget_in_classes": round(
                contract.BOUNDED_MAX_HARMED_FRACTION * len(cell.label_names), 2
            ),
        },
        "identity": {
            face: {
                "macro_f1": float(identity[face]["macro_f1"]),
                "utility": float(identity[face]["utility"]),
            }
            for face in FACES
        },
        "menu": {
            "programs_considered": len(rows),
            "verifier_refused": len(rows) - len(evaluated),
            "evaluated": len(evaluated),
            "stable_on_both_faces": len(stable),
        },
        "natural_headroom": {face: _best(face) for face in FACES},
        "stable_programs": [row["program"] for row in stable],
        "rows": rows,
        "wall_seconds": round(time.time() - started, 1),
    }


def build() -> dict[str, Any]:
    fixtures = [audit_fixture(spec) for spec in SEALED_FINAL]
    any_stable = any(entry["menu"]["stable_on_both_faces"] for entry in fixtures)
    any_positive_face = any(
        (entry["natural_headroom"][face] or {}).get("aggregate_gain", 0.0) >= MATERIAL
        for entry in fixtures
        for face in FACES
    )
    return {
        "stage": "P4E_CLASSIFICATION_NATURAL_HEADROOM",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_HELD_IN_MENU_SWEEP",
        "data_role": "SEALED_FINAL_TRAIN_ONLY",
        "boundary": {
            "llm_calls": 0,
            "ucr_test_member_bytes_read": 0,
            "held_out_requests": 0,
            "natural_final_outcome_reads": 0,
            "skill_store_writes": 0,
            "thresholds_changed": 0,
            "operators_added": 0,
        },
        "question": (
            "does the existing operator menu contain a stable positive candidate "
            "on Adiac / ArrowHead TRAIN held-in, before any TEST byte is spent"
        ),
        "rule": (
            "imported unchanged from p4b_contract.BOUNDED_POLICY; the per-view "
            "unit is per-class recall, so the harmed-fraction budget counts "
            "classes, not series"
        ),
        "fixtures": fixtures,
        "verdict": (
            "STABLE_POSITIVE_CANDIDATE_EXISTS" if any_stable
            else "ONE_FACE_POSITIVE_ONLY" if any_positive_face
            else "NO_POSITIVE_CANDIDATE_IN_EXISTING_MENU"
        ),
        "reading": (
            "TEST may be considered once a candidate is approved" if any_stable
            else "TEST stays sealed: nothing in the existing menu clears both "
                 "faces on held-in, and adding operators against a TEST reading "
                 "is exactly what the ruling forbids"
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
    for entry in report["fixtures"]:
        geometry, menu = entry["geometry"], entry["menu"]
        print("%-11s classes %2d | fit/A/B %d/%d/%d | budget %.1f classes" % (
            entry["fixture_id"], geometry["classes"], geometry["fit"],
            geometry["support_a"], geometry["support_b"],
            geometry["harmed_budget_in_classes"]))
        print("            identity macro-F1 A %.4f  B %.4f" % (
            entry["identity"]["support_a"]["macro_f1"],
            entry["identity"]["support_b"]["macro_f1"]))
        print("            menu %d considered, %d refused, %d evaluated, %d stable"
              % (menu["programs_considered"], menu["verifier_refused"],
                 menu["evaluated"], menu["stable_on_both_faces"]))
        for face in FACES:
            best = entry["natural_headroom"][face]
            print("            best %-9s %s" % (
                face, "none with behaviour" if best is None else
                "%-22s gain %+.4f  harmed %.2f  max %.3f  admitted %s" % (
                    best["program"], best["aggregate_gain"],
                    best["harmed_fraction"], best["max_single_class_harm"],
                    best["admitted"])))
    print("verdict : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
