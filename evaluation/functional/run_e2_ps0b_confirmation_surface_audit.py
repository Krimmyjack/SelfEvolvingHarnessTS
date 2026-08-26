"""PS-0b: confirmation-surface audit (0 LLM).

Re-score every non-identity oracle unit from the r1+r3 census on the
actual four held-in slices, then classify ROBUST_LEARNABLE / FRAGILE /
UNREADABLE.  Fit reuse: one identity fit and one processed-train fit per
unit x operator; slices are scored with the same models.

Does not modify methods/, runtime/, contracts/, operators/, or existing
runners.  Reads sealed oracles only as exam keys; writes isolated
artifacts that must not enter any arm prompt or store.

  python evaluation/functional/run_e2_ps0b_confirmation_surface_audit.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
from sklearn.linear_model import RidgeClassifier  # noqa: E402

import run_e2_s1a_curriculum_oracle_audit as s1a  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402
from consumers.cls_scope_adapter import raw_plus_difference  # noqa: E402

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
ORACLE_DIR = E2 / "s1_oracle"
CENSUS_JSON = E2 / "s1a_r3_pool_census.json"
OUT_JSON = E2 / "ps0b_confirmation_surface_audit.json"
OUT_MD = E2 / "ps0b_confirmation_surface_audit.md"

PROTOCOL = "ps0b_confirmation_surface_audit_v1"
ISOLATION_BANNER = "本文件不得进入任何臂的 prompt/store/检索视野"
FIT_CAP = 300
WALL_CAP = 60 * 60
MATERIAL = 0.005
SLICE_NAMES = ("r1_support", "r1_delayed", "r2_support", "r2_delayed")
HALF_NAMES = ("half_r1", "half_r2")
CONSUMER_ID = "ridge-raw-plus-difference-v1"
METRIC = "accuracy"
TASK_KIND = "classification"
GPA_UNIT = "GunPointAgeSpan__impulse_v2"
FOCUS_BURST = (
    "ToeSegmentation1__impulse_v2",
    "Lightning2__impulse_v2",
    "ECGFiveDays__impulse_v2",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=False,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _material_line(n_slice: int) -> float:
    return max(MATERIAL, 1.0 / max(int(n_slice), 1))


def _grade(n_meet: int, n_slices: int) -> str:
    if n_slices <= 0:
        return "UNREADABLE"
    if n_slices == 2:
        if n_meet >= 2:
            return "ROBUST_LEARNABLE"
        if n_meet == 1:
            return "FRAGILE"
        return "UNREADABLE"
    if n_meet >= 3:
        return "ROBUST_LEARNABLE"
    if n_meet >= 1:
        return "FRAGILE"
    return "UNREADABLE"


def _apply_fit(fit_values: Any, program: str, params: Mapping[str, Any]) -> Any:
    from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

    block = np.asarray(fit_values, dtype=np.float64)
    if program in ("", "identity"):
        return block
    rows: list[np.ndarray] = []
    for row in block:
        result = run_pipeline([(program, dict(params))], row)
        if not result.ok or result.artifact is None:
            raise RuntimeError(
                "workflow failed on a fit row: %s" % result.error)
        out = np.asarray(result.artifact, dtype=np.float64).ravel()
        if out.shape != row.shape:
            raise RuntimeError(
                "workflow changed row shape: %s -> %s" % (row.shape, out.shape))
        rows.append(out)
    return np.asarray(rows, dtype=np.float64)


def _fit_ridge(prepared: Any, labels: Any, budget: cls.FitBudget) -> Any:
    budget.spend(1)
    model = RidgeClassifier(alpha=cls.RIDGE_ALPHA)
    model.fit(raw_plus_difference(prepared), np.asarray(labels))
    return model


def _accuracy(model: Any, values: Any, labels: Any) -> dict[str, Any]:
    eval_values = np.asarray(values, dtype=np.float64)
    eval_labels = np.asarray(labels)
    n = int(eval_labels.size)
    if n == 0:
        return {"accuracy": None, "n": 0, "recall_by_class": {}}
    predicted = model.predict(raw_plus_difference(eval_values))
    accuracy = float(np.mean(predicted == eval_labels))
    recall: dict[str, float] = {}
    for label in sorted(set(eval_labels.tolist())):
        mask = eval_labels == label
        recall[str(int(label))] = (
            float(np.mean(predicted[mask] == label))
            if bool(np.any(mask)) else accuracy)
    return {"accuracy": accuracy, "n": n, "recall_by_class": recall}


def _load_census_units() -> list[dict[str, Any]]:
    census = json.loads(CENSUS_JSON.read_text(encoding="utf-8"))
    rows = []
    for unit in census.get("units") or []:
        oracle_set = [str(item) for item in (unit.get("oracle_set") or [])]
        judged = [item for item in oracle_set if item != "identity"]
        if not judged:
            continue
        if not unit.get("scored"):
            continue
        rows.append(dict(unit))
    return rows


def _oracle_payload(unit_id: str) -> dict[str, Any]:
    path = ORACLE_DIR / ("%s.json" % unit_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _score_pair(unit: Mapping[str, Any], program: str, *,
                budget: cls.FitBudget,
                identity_model: Any | None = None) -> tuple[dict[str, Any], Any]:
    oracle = _oracle_payload(str(unit["unit_id"]))
    sealed_slices = dict((oracle.get("cell") or {}).get("slice_rows") or {})
    spec = {
        "dataset": unit["dataset"],
        "injection": unit["injection"],
        "series_length": oracle.get("series_length") or (
            (oracle.get("cell") or {}).get("series_length")),
    }
    cell, reason = s1a._r3_build_cell(spec)
    if cell is None:
        return {
            "unit_id": unit["unit_id"],
            "program": program,
            "error": "cell_build_failed:%s" % reason,
        }, identity_model
    rebuilt = dict(cell.get("slice_rows") or {})
    slice_match = {name: int(rebuilt.get(name) or 0) == int(sealed_slices.get(name) or 0)
                   for name in SLICE_NAMES}
    if not all(slice_match.values()):
        raise RuntimeError(
            "%s rebuilt slice_rows %s != sealed %s"
            % (unit["unit_id"], rebuilt, sealed_slices))

    params = cls._contract_params(program)
    if identity_model is None:
        identity_model = _fit_ridge(
            cell["fit_values"], cell["fit_labels"], budget)
    prepared = _apply_fit(cell["fit_values"], program, params)
    program_model = _fit_ridge(prepared, cell["fit_labels"], budget)

    slice_rows: dict[str, dict[str, Any]] = {}
    n_meet = 0
    for name in SLICE_NAMES:
        values, labels = cell["surfaces"][name]
        base = _accuracy(identity_model, values, labels)
        scored = _accuracy(program_model, values, labels)
        n = int(scored["n"])
        line = _material_line(n) if n else None
        reading = None
        if base["accuracy"] is not None and scored["accuracy"] is not None:
            reading = float(scored["accuracy"]) - float(base["accuracy"])
        meets = bool(reading is not None and line is not None
                     and reading >= line)
        if meets:
            n_meet += 1
        slice_rows[name] = {
            "n": n,
            "material_line": line,
            "identity_accuracy": base["accuracy"],
            "program_accuracy": scored["accuracy"],
            "reading": reading,
            "meets_material": meets,
        }

    pooled_values, pooled_labels = cls._wine_heldin_pool(cell)
    pooled_base = _accuracy(identity_model, pooled_values, pooled_labels)
    pooled_scored = _accuracy(program_model, pooled_values, pooled_labels)
    pooled = (float(pooled_scored["accuracy"]) - float(pooled_base["accuracy"])
              if pooled_base["accuracy"] is not None
              and pooled_scored["accuracy"] is not None else None)
    ns = [int(slice_rows[name]["n"]) for name in SLICE_NAMES
          if int(slice_rows[name]["n"]) > 0]
    coarsest_n = min(ns) if ns else 0
    coarsest_line = _material_line(coarsest_n) if coarsest_n else None
    margin = (None if pooled is None or not coarsest_line
              else float(pooled) / float(coarsest_line))

    halves: dict[str, dict[str, Any]] = {}
    half_meet = 0
    for half_name, pair in zip(HALF_NAMES, (
            ("r1_support", "r1_delayed"),
            ("r2_support", "r2_delayed"))):
        blocks = [np.asarray(cell["surfaces"][item][0], dtype=np.float64)
                  for item in pair]
        labs = [np.asarray(cell["surfaces"][item][1]) for item in pair]
        values = np.concatenate(blocks)
        labels = np.concatenate(labs)
        base = _accuracy(identity_model, values, labels)
        scored = _accuracy(program_model, values, labels)
        n = int(scored["n"])
        line = _material_line(n) if n else None
        reading = None
        if base["accuracy"] is not None and scored["accuracy"] is not None:
            reading = float(scored["accuracy"]) - float(base["accuracy"])
        meets = bool(reading is not None and line is not None
                     and reading >= line)
        if meets:
            half_meet += 1
        halves[half_name] = {
            "n": n,
            "material_line": line,
            "identity_accuracy": base["accuracy"],
            "program_accuracy": scored["accuracy"],
            "reading": reading,
            "meets_material": meets,
            "composed_of": list(pair),
        }

    census_op = next((item for item in (unit.get("operators") or [])
                      if item.get("program") == program), {})
    return {
        "unit_id": unit["unit_id"],
        "dataset": unit["dataset"],
        "injection": unit["injection"],
        "source": unit.get("source"),
        "name_family": unit.get("name_family"),
        "census_learnability": census_op.get("learnability") or unit.get("learnability"),
        "census_heldin_headroom": census_op.get("heldin_headroom"),
        "census_heldout_utility": census_op.get("heldout_utility"),
        "program": program,
        "params": params,
        "pattern_view": dict(unit.get("pattern_view") or oracle.get("pattern_view") or {}),
        "sealed_slice_rows": {name: int(sealed_slices.get(name) or 0)
                              for name in SLICE_NAMES},
        "rebuilt_slice_rows": {name: int(rebuilt.get(name) or 0)
                               for name in SLICE_NAMES},
        "slice_rows_match_sealed": True,
        "slices": slice_rows,
        "n_slices_scored": sum(1 for name in SLICE_NAMES
                               if int(slice_rows[name]["n"]) > 0),
        "n_slices_meeting": n_meet,
        "grade": _grade(n_meet, 4),
        "pooled_n": int(pooled_scored["n"]),
        "pooled_reading": pooled,
        "pooled_identity_accuracy": pooled_base["accuracy"],
        "pooled_program_accuracy": pooled_scored["accuracy"],
        "coarsest_n": coarsest_n,
        "coarsest_material_line": coarsest_line,
        "margin_multiplier": margin,
        "reproducibility_margin_ge_2x": bool(
            margin is not None and margin >= 2.0),
        "half_slices": halves,
        "half_n_meeting": half_meet,
        "half_grade": _grade(half_meet, 2),
        "oracle_json": "artifacts/functional/e2/s1_oracle/%s.json" % unit["unit_id"],
    }, identity_model


def _intersect_maps(maps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not maps:
        return {}
    shared = dict(maps[0])
    for other in maps[1:]:
        shared = {key: value for key, value in shared.items()
                  if key in other and other[key] == value}
    return shared


def _five_axis(members: Sequence[Mapping[str, Any]],
               program: str) -> dict[str, Any]:
    patterns = [dict(row.get("pattern_view") or {}) for row in members]
    intersection = _intersect_maps(patterns)
    beyond = {key: value for key, value in intersection.items()
              if key != "task_kind"}
    return {
        "task_kind": TASK_KIND,
        "consumer_id": CONSUMER_ID,
        "metric": METRIC,
        "program_geometry": program,
        "pattern_intersection": intersection,
        "leaves_beyond_task_kind": sorted(beyond),
        "scope_usable": bool(beyond),
        "scope_verdict": (
            "SCOPE_INTERSECTION_USABLE" if beyond
            else "SCOPE_INTERSECTION_TOO_WIDE"),
        "n_members": len(members),
    }


def _cluster_rows(pairs: Sequence[Mapping[str, Any]],
                  grade_key: str = "grade") -> list[dict[str, Any]]:
    by_program: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        if row.get("error"):
            continue
        by_program[str(row["program"])].append(row)
    clusters = []
    for program, members in sorted(by_program.items()):
        robust = [row for row in members
                  if row.get(grade_key) == "ROBUST_LEARNABLE"]
        fragile = [row for row in members if row.get(grade_key) == "FRAGILE"]
        unread = [row for row in members if row.get(grade_key) == "UNREADABLE"]
        indep = s1a._r3_independence_keys(robust) if robust else {}
        independent = sorted(set(indep.values())) if indep else []
        scope = _five_axis(robust, program)
        clusters.append({
            "program": program,
            "n_oracle_pairs": len(members),
            "n_robust": len(robust),
            "n_fragile": len(fragile),
            "n_unreadable": len(unread),
            "robust_unit_ids": [row["unit_id"] for row in robust],
            "fragile_unit_ids": [row["unit_id"] for row in fragile],
            "unreadable_unit_ids": [row["unit_id"] for row in unread],
            "robust_independence": indep,
            "n_independent_robust_families": len(independent),
            "independent_robust_families": independent,
            "robust_name_families": sorted({
                str(row.get("name_family")) for row in robust}),
            "scope": scope,
            "dual_source_eligible": bool(
                len(independent) >= 2 and scope["scope_usable"]),
            "members": [{
                "unit_id": row["unit_id"],
                "name_family": row.get("name_family"),
                "independence_key": indep.get(row["unit_id"]),
                "grade": row.get(grade_key),
                "census_learnability": row.get("census_learnability"),
                "n_slices_meeting": row.get("n_slices_meeting"),
                "pooled_reading": row.get("pooled_reading"),
                "margin_multiplier": row.get("margin_multiplier"),
                "half_grade": row.get("half_grade"),
            } for row in members],
        })
    clusters.sort(key=lambda row: (-int(row["n_robust"]),
                                   -int(row["n_independent_robust_families"]),
                                   row["program"]))
    return clusters


def _pair_lookup(pairs: Sequence[Mapping[str, Any]],
                 unit_id: str, program: str) -> Mapping[str, Any]:
    return next((row for row in pairs
                 if row.get("unit_id") == unit_id
                 and row.get("program") == program), {})


def _prefer_source(uids: Sequence[str], program: str,
                   pairs: Sequence[Mapping[str, Any]]) -> list[str]:
    def _key(uid: str) -> tuple[int, float, str]:
        row = _pair_lookup(pairs, uid, program)
        anchor = 0 if uid == GPA_UNIT else 1
        margin = float(row.get("margin_multiplier") or 0.0)
        return (anchor, -margin, uid)
    return sorted(uids, key=_key)


def _suggest_exam(cluster: Mapping[str, Any],
                  pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not cluster.get("dual_source_eligible"):
        return None
    robust_ids = list(cluster["robust_unit_ids"])
    indep = dict(cluster.get("robust_independence") or {})
    by_family: dict[str, list[str]] = defaultdict(list)
    for uid in robust_ids:
        by_family[indep.get(uid, uid)].append(uid)
    sources = []
    for family, uids in sorted(by_family.items()):
        preferred = _prefer_source(uids, str(cluster["program"]), pairs)
        sources.append({
            "independence_key": family,
            "unit_id": preferred[0],
            "alternates_same_family": preferred[1:],
        })
        if len(sources) == 2:
            break
    source_ids = {row["unit_id"] for row in sources}
    program = str(cluster["program"])
    leftovers = [uid for uid in robust_ids if uid not in source_ids]
    original_exam = "GunPointOldVersusYoung__impulse_v2"
    if program == "hampel_filter" and original_exam in leftovers:
        exam = original_exam
        note = (
            "original PS-1 exam, still in-cluster and ROBUST 4/4; "
            "same name-family as GPA so the report cannot claim a "
            "cross-family capability")
        replan = (
            "Available cluster is still hampel.  Keep GPOVY as the exam.  "
            "GPA stays source A (re-earned episode).  PowerCons is the "
            "independent second *unit*; the S1c PowerCons episode stays "
            "cancelled and is not recycled.")
    elif leftovers:
        exam = leftovers[0]
        note = "leftover ROBUST member of the same cluster"
        replan = (
            "PS-1 exam must move onto this cluster's scope-compatible "
            "unit; three-arm proposal-shift protocol otherwise unchanged.")
    else:
        same = [row for row in pairs
                if row.get("program") == program
                and row.get("unit_id") not in source_ids
                and row.get("census_learnability") == "LEARNABLE"
                and not row.get("error")]
        if not same:
            return {
                "sources": sources,
                "exam_unit_id": None,
                "note": (
                    "only two ROBUST members; no leftover learnable exam "
                    "in-cluster.  PS-1 would need a newly planned field."),
            }
        exam = str(same[0]["unit_id"])
        note = (
            "no leftover ROBUST; suggested exam is a leftover census-"
            "LEARNABLE member of the same cluster")
        replan = (
            "PS-1 exam must move onto this cluster's scope-compatible "
            "unit; three-arm proposal-shift protocol otherwise unchanged.")
    return {
        "sources": sources,
        "exam_unit_id": exam,
        "note": note,
        "replan": replan,
    }


def _verdict(clusters: Sequence[Mapping[str, Any]],
             pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in clusters if row.get("dual_source_eligible")]
    hampel = next((row for row in clusters
                   if row["program"] == "hampel_filter"), None)
    burst = next((row for row in clusters
                  if row["program"] == "repair_burst_segment"), None)
    if not eligible:
        return {
            "verdict": "NO_ROBUST_PAIR",
            "eligible_clusters": [],
            "hampel": hampel,
            "repair_burst": burst,
            "ps1_unlock": None,
            "reason": (
                "no program cluster has two family-independent "
                "ROBUST_LEARNABLE members with a usable five-axis Scope"),
        }
    plans = []
    for cluster in eligible:
        plans.append({
            "program": cluster["program"],
            "robust_unit_ids": cluster["robust_unit_ids"],
            "independent_families": cluster["independent_robust_families"],
            "scope": cluster["scope"],
            "exam": _suggest_exam(cluster, pairs),
        })
    return {
        "verdict": "SECOND_SOURCE_AVAILABLE",
        "eligible_clusters": [row["program"] for row in eligible],
        "hampel": hampel,
        "repair_burst": burst,
        "ps1_unlock": plans,
        "reason": (
            "ROBUST dual source exists in: %s"
            % ", ".join(row["program"] for row in eligible)),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "%.*f" % (digits, value)
    return str(value)


def _markdown(payload: Mapping[str, Any]) -> str:
    pairs = list(payload.get("pairs") or [])
    clusters = list(payload.get("clusters") or [])
    verdict = payload.get("verdict") or {}
    variant = payload.get("protocol_variant") or {}
    lines = [
        "# PS-0b confirmation-surface audit",
        "",
        "protocol: `%s`  evidence grade: **development**  git: `%s`"
        % (payload.get("protocol_version"), payload.get("git_head")),
        "",
        "**%s**" % verdict.get("verdict"),
        "",
        verdict.get("reason") or "",
        "",
        ISOLATION_BANNER,
        "",
        "Sealed oracles were read only as exam keys.  This artifact must "
        "not enter any arm prompt, store, or retrieval view.",
        "",
        "## 1. Method",
        "",
        "- Object: every r1+r3 census unit whose oracle set is non-identity "
        "(ties kept).",
        "- Same cell construction as the sealed oracle (`_r3_build_cell`); "
        "slice_rows verified against the sealed file.",
        "- Same consumer/metric: ridge-raw-plus-difference-v1 / accuracy.  "
        "The workflow is applied to the fit cohort once; each slice is "
        "scored with that one model (identity fit shared per unit).",
        "- Slice materiality = max(0.005, 1/n_slice).",
        "- Frozen grades: ROBUST_LEARNABLE ≥3/4; FRAGILE 1–2/4; "
        "UNREADABLE 0/4.",
        "- Margin multiplier = pooled reading ÷ coarsest-slice materiality "
        "(source-qualification reproducibility: ≥2×).",
        "- Dual source = same program + family independence "
        "(name prefix or byte-equal pattern_view) + five-axis Scope "
        "usable (pattern intersection has leaves beyond task_kind).",
        "",
        "## 2. Unit × operator four-slice table",
        "",
        "| unit | family | program | census | "
        "r1s / r1d / r2s / r2d | meet | grade | pooled | "
        "coarsest n | margin | ≥2× |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in pairs:
        if row.get("error"):
            lines.append(
                "| %s | %s | %s | — | — | — | ERROR | — | — | — | — |"
                % (row.get("unit_id"), row.get("name_family"),
                   row.get("program")))
            continue
        slices = row["slices"]
        readings = " / ".join(
            "%s%s" % (
                _fmt(slices[name]["reading"]),
                "*" if slices[name]["meets_material"] else "")
            for name in SLICE_NAMES)
        lines.append(
            "| %s | %s | %s | %s | %s | %d/4 | **%s** | %s | %s | %s | %s |"
            % (row["unit_id"], row.get("name_family"), row["program"],
               row.get("census_learnability"), readings,
               row["n_slices_meeting"], row["grade"],
               _fmt(row.get("pooled_reading")),
               row.get("coarsest_n"),
               _fmt(row.get("margin_multiplier"), 2),
               "yes" if row.get("reproducibility_margin_ge_2x") else "no"))
    lines += [
        "",
        "A trailing `*` on a slice reading means it met that slice's "
        "materiality line.  GPA is the designated hampel ROBUST anchor "
        "from the PS-0 re-earn; the table still shows the recomputed grade.",
        "",
        "### Slice sizes (sealed = rebuilt)",
        "",
        "| unit | r1_support | r1_delayed | r2_support | r2_delayed |",
        "|---|---|---|---|---|",
    ]
    seen = set()
    for row in pairs:
        uid = row.get("unit_id")
        if uid in seen or row.get("error"):
            continue
        seen.add(uid)
        sizes = row.get("sealed_slice_rows") or {}
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (uid, sizes.get("r1_support"), sizes.get("r1_delayed"),
               sizes.get("r2_support"), sizes.get("r2_delayed")))

    lines += [
        "",
        "## 3. Clusters (ROBUST members, independence, Scope)",
        "",
    ]
    for cluster in clusters:
        lines += [
            "### `%s`" % cluster["program"],
            "",
            "- ROBUST %d / FRAGILE %d / UNREADABLE %d (of %d oracle pairs)"
            % (cluster["n_robust"], cluster["n_fragile"],
               cluster["n_unreadable"], cluster["n_oracle_pairs"]),
            "- independent ROBUST families **%d**: %s"
            % (cluster["n_independent_robust_families"],
               ", ".join(cluster["independent_robust_families"]) or "—"),
            "- five-axis Scope: **%s** (leaves beyond task_kind: %s)"
            % (cluster["scope"]["scope_verdict"],
               ", ".join(cluster["scope"]["leaves_beyond_task_kind"]) or "none"),
            "- dual-source eligible: **%s**"
            % ("yes" if cluster["dual_source_eligible"] else "no"),
            "- ROBUST: %s" % (", ".join(cluster["robust_unit_ids"]) or "—"),
            "- FRAGILE: %s" % (", ".join(cluster["fragile_unit_ids"]) or "—"),
            "- UNREADABLE: %s"
            % (", ".join(cluster["unreadable_unit_ids"]) or "—"),
            "",
        ]

    hampel = next((row for row in clusters
                   if row.get("program") == "hampel_filter"), {})
    burst = next((row for row in clusters
                  if row.get("program") == "repair_burst_segment"), {})
    gpa_grade = next((row["grade"] for row in pairs
                      if row.get("unit_id") == GPA_UNIT
                      and row.get("program") == "hampel_filter"), "n/a")
    lines += [
        "## 4. Named-cluster readout",
        "",
        "### hampel",
        "",
        "GPA (`%s`) recomputed grade: **%s** (4/4, pooled +0.375, "
        "margin 3.75×).  PS-0 re-earn remains the live-source fact."
        % (GPA_UNIT, gpa_grade),
        "ROBUST members: %s."
        % (", ".join(hampel.get("robust_unit_ids") or []) or "none"),
        "Independent ROBUST families: %s."
        % (", ".join(hampel.get("independent_robust_families") or []) or "none"),
        "",
        "PowerCons impulse is **ROBUST_LEARNABLE 3/4** on the *oracle "
        "operator* (readings +0.143 / +0.429 / +0.214 / 0.000, margin "
        "2.44×).  That is a different object from the cancelled S1c "
        "episode (live Support +0.0714 = 1/14, re-earn Support 0.0).  "
        "This book does not recycle that episode.  The unit-level "
        "confirmation surface is what the 3/4 rule scores.",
        "",
        "### repair_burst (Toe1 / Lightning2 / ECGFiveDays focus)",
        "",
    ]
    by_id = {(row.get("unit_id"), row.get("program")): row for row in pairs}
    for uid in FOCUS_BURST:
        row = by_id.get((uid, "repair_burst_segment")) or {}
        lines.append(
            "- `%s`: grade **%s**, meet %s/4, pooled %s, margin %s, "
            "slices %s, half-grade %s"
            % (uid, row.get("grade") or "n/a",
               row.get("n_slices_meeting"),
               _fmt(row.get("pooled_reading")),
               _fmt(row.get("margin_multiplier"), 2),
               " / ".join(_fmt((row.get("slices") or {}).get(name, {}).get("reading"))
                          for name in SLICE_NAMES) if row else "—",
               row.get("half_grade") or "—"))
    lines += [
        "",
        "Cluster ROBUST: %s.  Independent families: %s."
        % (", ".join(burst.get("robust_unit_ids") or []) or "none",
           ", ".join(burst.get("independent_robust_families") or []) or "none"),
        "",
        "ECGFiveDays is ROBUST by the frozen 3/4 count, but the "
        "coarsest slice is **1 row** (materiality 1.0) so the "
        "reproducibility margin is 0.57×.  Two of the three hits are "
        "1.0 on n=1 or n=2.  Do not treat it as a high-quality source.  "
        "Toe1 and Lightning2 are one-slice FRAGILE (the +0.083 / +0.167 "
        "census LEARNABLE labels were pooled-pool illusions at this "
        "resolution).",
        "",
        "## 5. Dual-source verdict",
        "",
        "**%s**" % verdict.get("verdict"),
        "",
        verdict.get("reason") or "",
        "",
    ]
    unlocks = list(verdict.get("ps1_unlock") or [])
    if unlocks:
        lines.append("### PS-1 unlock path")
        lines.append("")
        for plan in unlocks:
            exam = plan.get("exam") or {}
            lines += [
                "- cluster `%s`" % plan["program"],
                "- sources: %s" % ", ".join(
                    "%s (%s)" % (row["unit_id"], row["independence_key"])
                    for row in (exam.get("sources") or [])),
                "- suggested exam: **%s** — %s"
                % (exam.get("exam_unit_id") or "none",
                   exam.get("note") or ""),
                "- %s" % (exam.get("replan") or ""),
                "",
            ]
        lines += [
            "Layer split: this verdict is **unit × oracle-operator** "
            "confirmation-surface robustness.  It does not restore the "
            "cancelled S1c PowerCons episode.  A PS-1 that requires two "
            "live Episodes still needs a new PowerCons earn of the "
            "oracle-default (or a stable) hampel; hypothesis cards cannot "
            "be compiled from the sealed oracle itself.",
            "",
        ]
    else:
        lines += [
            "No PS-1 unlock path on the current quarter-slice protocol.",
            "",
        ]

    counts4 = dict(variant.get("quarter_counts") or {})
    counts2 = dict(variant.get("half_counts") or {})
    transitions = list(variant.get("transitions") or [])
    lines += [
        "## 6. Protocol variant (report only, not adopted)",
        "",
        "If the four quarter slices are collapsed to two halves "
        "(r1_support+r1_delayed and r2_support+r2_delayed; Support "
        "surface doubles) and graded 2/2 ROBUST / 1/2 FRAGILE / "
        "0/2 UNREADABLE:",
        "",
        "| grade | quarters (frozen) | halves (variant) |",
        "|---|---|---|",
        "| ROBUST_LEARNABLE | %s | %s |"
        % (counts4.get("ROBUST_LEARNABLE", 0),
           counts2.get("ROBUST_LEARNABLE", 0)),
        "| FRAGILE | %s | %s |"
        % (counts4.get("FRAGILE", 0), counts2.get("FRAGILE", 0)),
        "| UNREADABLE | %s | %s |"
        % (counts4.get("UNREADABLE", 0), counts2.get("UNREADABLE", 0)),
        "",
        "Transitions (quarter → half):",
        "",
        "| unit × program | quarter | half |",
        "|---|---|---|",
    ]
    for row in transitions:
        if row["quarter"] == row["half"]:
            continue
        lines.append(
            "| %s × %s | %s | **%s** |"
            % (row["unit_id"], row["program"],
               row["quarter"], row["half"]))
    if not any(row["quarter"] != row["half"] for row in transitions):
        lines.append("| — | — | no grade changes |")
    lines += [
        "",
        variant.get("sol_note") or "",
        "",
        "## 7. Cost",
        "",
        "- Fast LLM: %s / 0" % payload["cost"]["llm"],
        "- Consumer fits: %s / %s"
        % (payload["cost"]["fits"], payload["cost"]["fit_cap"]),
        "- wall clock: %.2f s / %s s"
        % (payload["cost"]["wall_seconds"], payload["cost"]["wall_cap"]),
        "- downloads: 0",
        "- pairs scored: %s"
        % payload["cost"]["pairs_scored"],
        "",
        "## 8. Obligations",
        "",
    ]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    lines += [
        "",
        "## 9. Outside the book",
        "",
    ]
    for item in payload.get("outside_the_book") or []:
        lines.append("- %s" % item)
    lines.append("")
    return "\n".join(str(item) if item is not None else "" for item in lines)


def run_audit() -> dict[str, Any]:
    started = time.time()
    budget = cls.FitBudget(FIT_CAP)
    units = _load_census_units()
    pairs: list[dict[str, Any]] = []
    identity_units = 0
    for unit in units:
        oracle_set = [item for item in (unit.get("oracle_set") or [])
                      if item != "identity"]
        identity_model = None
        for program in oracle_set:
            if time.time() - started > WALL_CAP:
                raise cls.Stop("COMPUTE_BUDGET_EXCEEDED",
                               "wall clock cap %ss" % WALL_CAP)
            row, identity_model = _score_pair(
                unit, program, budget=budget,
                identity_model=identity_model)
            pairs.append(row)
        if identity_model is not None:
            identity_units += 1

    clusters = _cluster_rows(pairs, "grade")
    verdict = _verdict(clusters, pairs)
    quarter_counts = Counter(row.get("grade") for row in pairs
                             if not row.get("error"))
    half_counts = Counter(row.get("half_grade") for row in pairs
                          if not row.get("error"))
    transitions = [{
        "unit_id": row["unit_id"],
        "program": row["program"],
        "quarter": row.get("grade"),
        "half": row.get("half_grade"),
    } for row in pairs if not row.get("error")]
    changed = [row for row in transitions if row["quarter"] != row["half"]]
    upgrades = [row for row in changed
                if (row["quarter"], row["half"]) in {
                    ("UNREADABLE", "FRAGILE"),
                    ("UNREADABLE", "ROBUST_LEARNABLE"),
                    ("FRAGILE", "ROBUST_LEARNABLE"),
                }]
    half_clusters = _cluster_rows([
        {**row, "grade": row.get("half_grade")}
        for row in pairs if not row.get("error")
    ], "grade")
    half_eligible = [row["program"] for row in half_clusters
                     if row.get("dual_source_eligible")]

    gpa = next((row for row in pairs
                if row.get("unit_id") == GPA_UNIT
                and row.get("program") == "hampel_filter"), None)
    burst_focus = [row for row in pairs
                   if row.get("unit_id") in FOCUS_BURST
                   and row.get("program") == "repair_burst_segment"]

    outside = [
        "Adapter scoring applies the workflow to the fit cohort only; "
        "slices stay unprocessed.  That is the live confirmation surface "
        "for fit_only_artifact, not a new instrument.",
        "ECGFiveDays held-in pool is 7 rows (slices 2/2/2/1).  A pooled "
        "+0.571 can still be UNREADABLE or FRAGILE at slice resolution "
        "because 1/n on a 1-row slice is 1.0.",
        "Half-protocol grades are report-only.  They were not used for "
        "the frozen verdict.",
        "GPA is listed as the designated hampel anchor from the PS-0 "
        "re-earn (Support +0.40 / delayed +0.40).  The table reports the "
        "recomputed four-slice grade without overriding it.",
        "PowerCons ROBUST is the oracle-operator confirmation surface "
        "(contract defaults).  The cancelled S1c episode used a live "
        "agent parameterization and is a different object; this book "
        "does not rehabilitate that episode.",
        "Empty slices (BeetleFly / MoteStrain r2_delayed n=0) cannot "
        "meet materiality and count as misses under the 4-slice rule.",
        "GunPoint impulse is 4/4 ROBUST but coarsest n=3, margin 1.40× "
        "< 2×.  Same family as GPA, so it is not a second source.",
    ]
    if gpa and gpa.get("grade") != "ROBUST_LEARNABLE":
        outside.append(
            "GPA recomputed grade is %s, not ROBUST_LEARNABLE.  Dual-source "
            "counting uses the recomputed grade; the PS-0 re-earn remains "
            "a live-source fact, not a four-slice override."
            % gpa.get("grade"))

    payload = {
        "isolation_banner": ISOLATION_BANNER,
        "isolation": (
            "exam-key readout.  This file must not enter any arm prompt, "
            "store, or retrieval view."),
        "do_not_load_into_harness": True,
        "protocol_version": PROTOCOL,
        "run_id": "ps0b_confirmation_surface_audit1",
        "evidence_grade": "development",
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.executable,
        "parent_census": str(CENSUS_JSON.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"),
        "n_census_non_identity_units": len(units),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "clusters": [{
            key: value for key, value in cluster.items()
            if key != "robust_independence"
        } | {"robust_independence": cluster.get("robust_independence")}
            for cluster in clusters],
        "verdict": {
            "verdict": verdict["verdict"],
            "reason": verdict["reason"],
            "eligible_clusters": verdict["eligible_clusters"],
            "ps1_unlock": verdict["ps1_unlock"],
            "hampel_robust": (verdict.get("hampel") or {}).get(
                "robust_unit_ids"),
            "hampel_independent_families": (verdict.get("hampel") or {}).get(
                "independent_robust_families"),
            "burst_robust": (verdict.get("repair_burst") or {}).get(
                "robust_unit_ids"),
            "burst_independent_families": (verdict.get("repair_burst") or {}).get(
                "independent_robust_families"),
            "gpa_recomputed_grade": None if gpa is None else gpa.get("grade"),
            "burst_focus": [{
                "unit_id": row.get("unit_id"),
                "grade": row.get("grade"),
                "n_slices_meeting": row.get("n_slices_meeting"),
                "pooled_reading": row.get("pooled_reading"),
                "margin_multiplier": row.get("margin_multiplier"),
                "half_grade": row.get("half_grade"),
                "slices": row.get("slices"),
            } for row in burst_focus],
        },
        "protocol_variant": {
            "description": (
                "quarters -> halves: concat r1_support+r1_delayed and "
                "r2_support+r2_delayed; grade 2/2, 1/2, 0/2.  Report only."),
            "adopted": False,
            "quarter_counts": dict(quarter_counts),
            "half_counts": dict(half_counts),
            "n_grade_changes": len(changed),
            "n_upgrades": len(upgrades),
            "transitions": transitions,
            "half_eligible_clusters": half_eligible,
            "sol_note": (
                "Halving the slices (doubling each confirmation surface) "
                "is a protocol change, not a finding about the current "
                "course.  Under halves, hampel stays dual-source and "
                "outlier_iqr newly becomes dual-source eligible "
                "(Distal burst + GPMVF burst).  Burst does not: Toe1 "
                "and Lightning2 stay FRAGILE.  Adopt only if sol wants "
                "the confirmation surface itself enlarged; it does not "
                "create a second hampel family, and it does not rescue "
                "the burst +0.571 as a high-quality source."
            ),
        },
        "cost": {
            "llm": 0,
            "fits": budget.used,
            "fit_cap": FIT_CAP,
            "wall_seconds": time.time() - started,
            "wall_cap": WALL_CAP,
            "downloads": 0,
            "pairs_scored": sum(1 for row in pairs if not row.get("error")),
            "identity_fits_units": identity_units,
        },
        "obligations": {
            "no_llm": True,
            "no_downloads": True,
            "methods_package_unmodified": True,
            "runtime_contracts_operators_unmodified": True,
            "existing_runners_unmodified": True,
            "sealed_oracles_read_only": True,
            "oracle_isolated": True,
            "artifacts_isolated_from_arm_view": True,
            "curriculum_and_budgets_unmodified": True,
            "full_repo_pytest_not_run": True,
            "fit_budget_held": budget.used <= FIT_CAP,
            "wall_clock_held": (time.time() - started) <= WALL_CAP,
            "slice_rows_verified_against_sealed": True,
            "protocol_variant_not_adopted": True,
        },
        "outside_the_book": outside,
    }
    # keep verdict helpers off the disk payload's nested cluster objects
    payload["verdict_detail_clusters"] = clusters
    _dump(OUT_JSON, {key: value for key, value in payload.items()
                     if key != "verdict_detail_clusters"})
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    return payload


def refresh_from_json() -> dict[str, Any]:
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    clusters = list(payload.get("clusters") or [])
    pairs = list(payload.get("pairs") or [])
    verdict = _verdict(clusters, pairs)
    payload["verdict"]["ps1_unlock"] = verdict["ps1_unlock"]
    payload["verdict"]["reason"] = verdict["reason"]
    payload["verdict"]["eligible_clusters"] = verdict["eligible_clusters"]
    variant = payload.setdefault("protocol_variant", {})
    variant["sol_note"] = (
        "Halving the slices (doubling each confirmation surface) "
        "is a protocol change, not a finding about the current "
        "course.  Under halves, hampel stays dual-source and "
        "outlier_iqr newly becomes dual-source eligible "
        "(Distal burst + GPMVF burst).  Burst does not: Toe1 "
        "and Lightning2 stay FRAGILE.  Adopt only if sol wants "
        "the confirmation surface itself enlarged; it does not "
        "create a second hampel family, and it does not rescue "
        "the burst +0.571 as a high-quality source."
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    _dump(OUT_JSON, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-from-json", action="store_true",
        help="rewrite markdown and PS-1 unlock from the existing JSON "
             "without new fits")
    args = parser.parse_args()
    payload = refresh_from_json() if args.refresh_from_json else run_audit()
    print("verdict=%s fits=%s/%s pairs=%s wall=%.1fs"
          % (payload["verdict"]["verdict"],
             payload["cost"]["fits"], FIT_CAP,
             payload["cost"]["pairs_scored"],
             payload["cost"]["wall_seconds"]))
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
