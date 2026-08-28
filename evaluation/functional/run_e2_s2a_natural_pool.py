"""S2a Part N/O: #31 natural-pool remount + course branch.

0 LLM / 0 fit. Old #31 readings locate the pool only; they are not evidence.
Capacity gate is the current forecast cell rule (TRAIN>=40, half>=20).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_e2_s2a_forecast_oracle as traffic  # noqa: E402

PROTOCOL = "s2a_natural_pool_sweep_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "s2a_natural_pool_sweep.json"
OUT_MD = E2 / "s2a_natural_pool_sweep.md"
COURSE_JSON = E2 / "s2a_course_frozen.json"
COURSE_MD = E2 / "s2a_course_frozen.md"
CANDIDATE_PROGRAMS = ("winsorize", "outlier_mad")
N_TRAIN_GATE = 40
N_FACE_GATE = 20


def _pool_members() -> list[dict[str, Any]]:
    """The 12 deduped #31 episodes.  Cite the compiler and the v1 artifact."""
    # run_e2_shared_capability_candidate.py:8-17 / :51-100 traffic
    # :103-147 noaa; surviving list in
    # artifacts/functional/e2/shared_capability_candidate_v1.md:17
    traffic_eps = [
        "traffic/pooled/hampel_filter",
        "traffic/pooled/outlier_iqr",
        "traffic/pooled/outlier_mad",
        "traffic/pooled/winsorize",
        "traffic/per_channel/hampel_filter",
        "traffic/per_channel/outlier_iqr",
        "traffic/per_channel/outlier_mad",
        "traffic/per_channel/winsorize",
    ]
    noaa_eps = [
        "noaa/task_A/outlier_mad",
        "noaa/task_C/outlier_mad",
        "noaa/task_A/outlier_iqr",
        "noaa/task_D/outlier_mad",
    ]
    hidden = {
        "traffic/pooled/hampel_filter": ["14", "16", "17"],
        "traffic/pooled/outlier_iqr": ["14"],
        "noaa/task_C/outlier_mad": ["99999904140"],
        "noaa/task_A/outlier_iqr": ["99999923908"],
        "noaa/task_D/outlier_mad": ["99999904140", "99999963862"],
    }
    # C3 guard_rows: shared_capability_candidate_v1.json:822-880
    members = []
    for ep in traffic_eps:
        members.append({
            "episode": ep,
            "domain": "traffic",
            "n_train_original": 12,
            "n_eval_original": 8,
            "n_half_if_split": 6,
            "roster_cite": (
                "artifacts/functional/e2/batch_recipe_traffic_v1.json:16-38; "
                "run_batch_composition_headroom.py:163-164"
            ),
            "window_cite": (
                "batch_recipe_traffic_v1.json:9-15 support 1104/1368 "
                "delayed 1800"
            ),
            "hidden_harm_series": hidden.get(ep, []),
            "hidden_harm_cite": (
                "shared_capability_candidate_v1.json:822-846 (C3 guard_rows)"
            ),
        })
    for ep in noaa_eps:
        members.append({
            "episode": ep,
            "domain": "noaa",
            "n_train_original": 12,
            "n_eval_original": 4,
            "n_half_if_split": 6,
            "roster_cite": (
                "artifacts/functional/e2/fresh_confirmation_v1.json:64-65 "
                "ruling 12 train + 4 eval = 16 PASS stations"
            ),
            "window_cite": (
                "fresh_confirmation_v1.json WINDOWS task_A/C/D; "
                "run_e2_shared_capability_candidate.py:103-147"
            ),
            "hidden_harm_series": hidden.get(ep, []),
            "hidden_harm_cite": (
                "shared_capability_candidate_v1.json:848-880 (C3 guard_rows)"
            ),
        })
    return members


def _exclude(member: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if int(member["n_train_original"]) < N_TRAIN_GATE:
        reasons.append(
            "n_train=%d < %d" % (member["n_train_original"], N_TRAIN_GATE))
    if int(member["n_half_if_split"]) < N_FACE_GATE:
        reasons.append(
            "n_half=%d < %d" % (member["n_half_if_split"], N_FACE_GATE))
    # Hidden-harm series sit on the #31 eval/held-out face
    # (traffic 14/16/17 are eval_series in batch_recipe_traffic_v1.json:30-38).
    # Remounting without adding series cannot put them in a 20+20 half-split
    # of a 12-row train pool.  Adding series would be a new source.
    if member.get("hidden_harm_series"):
        reasons.append(
            "hidden-harm series are on the original eval face; "
            "cannot place them in Support/delayed of a TRAIN>=40 remount "
            "without adding series outside this pool member"
        )
    return {
        **member,
        "structural_exclusion": True,
        "exclusion_reasons": reasons,
        "remounted": False,
    }


def _injected_roles() -> dict[str, Any]:
    sweep = json.loads((E2 / "s2a_g0_electricity_sweep.json").read_text(
        encoding="utf-8"))
    cells = list((sweep.get("merged") or {}).get("headroom_table") or [])
    impulse = [c for c in cells
               if c.get("family") == "impulsive_outlier" and c.get("two_x")]
    ranked = sorted(impulse, key=lambda c: (-float(c["heldin_headroom"]),
                                            c["unit_id"]))
    producer = ranked[0]
    strong = ranked[1:3]
    t_names, _pool = traffic._load_pool()
    leftover = traffic.CELL_WIDTH * (
        traffic.N_IMPULSE_CELLS + traffic.N_GAP_CELLS)
    clean = list(t_names[leftover:leftover + traffic.CELL_WIDTH])
    return {
        "producer": producer,
        "strong": strong,
        "clean_chunk": clean,
        "ranked": ranked,
    }


def run() -> int:
    members = [_exclude(m) for m in _pool_members()]
    remounted = [m for m in members if m["remounted"]]
    table: list[dict[str, Any]] = []
    hits = [row for row in table if row.get("four_conjunction")]
    branch = "full" if hits else "reduced"
    roles = _injected_roles()
    if hits:
        raise RuntimeError("unexpected hit with zero remounts")
    course = [
        {"role": "producer", "unit_id": roles["producer"]["unit_id"],
         "source": "injected", "program_family_expected": "winsorize"},
        {"role": "clean_identity", "unit_id": "traffic_clean_identity_00",
         "source": "clean",
         "sol_name": "无缺陷条件下的 identity 场",
         "note": "clean condition cell; not a natural identity field"},
        {"role": "boundary_compile", "unit_id": roles["producer"]["unit_id"],
         "source": "injected",
         "note": "ladder v2 compile from producer strong positive; not a distinct cell"},
        {"role": "strong_beneficiary_1", "unit_id": roles["strong"][0]["unit_id"],
         "source": "injected"},
        {"role": "strong_beneficiary_2", "unit_id": roles["strong"][1]["unit_id"],
         "source": "injected"},
        {"role": "gap_out_of_family_guard", "unit_id": "traffic_gap_00",
         "source": "injected"},
    ]
    clean_cell = {
        "unit_id": "traffic_clean_identity_00",
        "dataset": "monash:traffic_hourly",
        "family": "clean",
        "sol_name": "无缺陷条件下的 identity 场",
        "train": roles["clean_chunk"][:traffic.N_TRAIN],
        "support": roles["clean_chunk"][:traffic.N_FACE],
        "delayed": roles["clean_chunk"][traffic.N_FACE:traffic.N_TRAIN],
        "heldout": roles["clean_chunk"][traffic.N_TRAIN:],
        "injection": "none",
        "source": "clean",
    }
    freeze = {
        "protocol": PROTOCOL,
        "branch": "reduced",
        "R2_forecast": "untested",
        "R2_note": (
            "R2 forecast 未考(冲突场在注入与 #31 自然池下均不可得)"
        ),
        "condition": traffic.CONDITION,
        "task_kind": traffic.TASK_KIND,
        "consumer_id": traffic.CONSUMER_ID,
        "metric": traffic.METRIC,
        "material_line": traffic._material_line(traffic.N_FACE),
        "delta_material": 2.0 * traffic._material_line(traffic.N_FACE),
        "delta_material_units": [
            roles["strong"][0]["unit_id"], roles["strong"][1]["unit_id"],
        ],
        "delta_material_note": (
            "sum of beneficiary-unit half-split material lines: "
            "0.05+0.05=0.10"
        ),
        "course": course,
        "clean_cell": clean_cell,
        "verdict_vocabulary": [
            "S2A_PORTABLE_REDUCED", "S2A_PARTIAL", "TREATMENT_EMPTY",
        ],
        "post_hoc_rejudge": (
            "If A5 live learns Q != expected family, re-judge any assigned "
            "conflict cell (none on the reduced course). Mark R2 untested."
        ),
        "oracle_isolation": traffic.ORACLE_BANNER,
    }
    payload = {
        "protocol": PROTOCOL,
        "branch": branch,
        "candidate_programs": list(CANDIDATE_PROGRAMS),
        "pool_cite": {
            "ledger": "docs/STAGE_REPORT #31 2026-08-22: 证据池去重 21→12 (traffic 8 + noaa 4)",
            "compiler": "evaluation/functional/run_e2_shared_capability_candidate.py:51-163",
            "artifact": "artifacts/functional/e2/shared_capability_candidate_v1.json/.md",
            "hidden_harms": "v1.json:818-880 C3 5/5 aggregate-hidden harms",
        },
        "old_31_readings_are_not_evidence": True,
        "members": members,
        "n_pool": len(members),
        "n_structurally_excluded": len(members),
        "n_remounted": len(remounted),
        "four_conjunction_table": table,
        "n_hits": 0,
        "fits": 0,
        "llm": 0,
        "course": course,
        "R2_note": freeze["R2_note"],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    COURSE_JSON.write_text(json.dumps(freeze, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    lines = [
        "# S2a #31 natural-pool sweep",
        "",
        "**branch: reduced (zero remounts, zero four-conjunction hits)**",
        "",
        "Pool = 12 #31 episodes (traffic 8 + noaa 4). "
        "Old #31 numbers locate the pool only.",
        "Capacity gate: TRAIN>=40, half>=20. All 12 fail (traffic 12+8, "
        "NOAA 12+4). Structural exclusion, not selection.",
        "Hidden-harm series (traffic 14/16/17; NOAA 99999904140 / 99999923908 / "
        "99999963862) sit on the original eval face and cannot enter a "
        "Support/delayed pool of a TRAIN>=40 remount without adding series.",
        "Fits: 0. Four-conjunction table: empty (no remounted cell to score).",
        "",
        "## Structural exclusions",
        "",
    ]
    for m in members:
        lines.append(
            "- `%s`: n_train=%d n_eval=%d hidden=%s reasons=%s"
            % (m["episode"], m["n_train_original"], m["n_eval_original"],
               ",".join(m["hidden_harm_series"]) or "none",
               "; ".join(m["exclusion_reasons"])))
    lines += [
        "",
        "## Reduced course",
        "",
        freeze["R2_note"],
        "",
    ]
    for row in course:
        lines.append("- %s: `%s` (%s)" % (row["role"], row["unit_id"],
                                          row["source"]))
    text = "\n".join(lines) + "\n"
    OUT_MD.write_text(text, encoding="utf-8")
    course_lines = [
        "# S2a reduced course freeze",
        "",
        "**branch: reduced. R2 forecast 未考(冲突场在注入与 #31 自然池下均不可得)**",
        "",
        "Zero four-conjunction hits after #31 remount attempt. "
        "sol pre-authorised the reduced shape. Defect source is labelled "
        "per cell. Verdict vocabulary: "
        "S2A_PORTABLE_REDUCED / S2A_PARTIAL / TREATMENT_EMPTY.",
        "",
        "Δ_material = 0.10 (two strong beneficiaries × 0.05).",
        "",
        "Live units (boundary is compile, not a cell):",
        "",
    ]
    for row in course:
        if row["role"] == "boundary_compile":
            course_lines.append(
                "- [boundary] ladder v2 compile on producer `%s` (%s)"
                % (row["unit_id"], row["source"]))
        else:
            course_lines.append(
                "- %s: `%s` (%s)%s"
                % (row["role"], row["unit_id"], row["source"],
                   " — 无缺陷条件下的 identity 场"
                   if row["role"] == "clean_identity" else ""))
    COURSE_MD.write_text("\n".join(course_lines) + "\n", encoding="utf-8")
    print("BRANCH reduced; remounted=0; hits=0", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.run:
        return run()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
