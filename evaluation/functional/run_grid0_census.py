"""run_grid0_census.py — GRID0 第 5 步资格 census（零 Outcome、零 LLM）。

协议：artifacts/functional/e2/_drafts/grid0_protocol.json rev2 + 主报告
grid0_protocol.rev3_notes_on_record。
本脚本不调用 ScopeExecutor.evaluate / nsu._evaluate_kdd，不计算任何 gain。
窗口 verifier 使用 ScopeExecutor.verify() 内部的同一原语 verify_candidate，
以获得逐窗口 modified_fraction 与修改点数（ScopeExecutor.verify 当前只返回
pass/reject，不暴露逐窗口比例，这是实现层事实，已记录在 checkpoint）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind  # noqa: E402
from SelfEvolvingHarnessTS.contracts.program import Program  # noqa: E402
from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate  # noqa: E402

CHECKPOINT_REL = PROJECT_ROOT / "artifacts" / "functional" / "e2" / "grid0_checkpoint.json"
PINNED_ANCHORS = (312, 372, 432, 492, 552)
ORIGINS = (600, 672, 744, 816, 888, 960)
CONTEXT = 192
HORIZON = 48
MAX_MODIFIED_FRACTION = 0.35
COHORT_A_N = 25
COHORT_B_N = 10
COHORT_A_EXCLUSIONS = {
    "T1", "T10", "T100", "T101",           # existing_dev
    "T13", "T128", "T129", "T130", "T131", "T132", "T133", "T134",  # sealed census eval
    "T117",                                 # previously_exposed
}
COHORT_B_ORDER = ("metr_la", "monash:traffic_hourly")


def natural_key(name: str) -> tuple[int, str]:
    m = re.match(r'^(?:T|MT_)?(\d+)$', str(name))
    if m:
        return (int(m.group(1)), str(name))
    return (10**9, str(name))


def _windows_ok(raw: np.ndarray) -> tuple[bool, str]:
    if not np.isfinite(raw).all():
        return False, "non_finite_values_present"
    n = raw.size
    needed = max(
        [a + HORIZON for a in PINNED_ANCHORS]
        + [o + HORIZON for o in ORIGINS]
    )
    start = min([a - CONTEXT for a in PINNED_ANCHORS]
                + [o - CONTEXT for o in ORIGINS])
    if start < 0 or needed > n:
        return False, f"window_out_of_bounds need_end={needed} len={n}"
    for a in PINNED_ANCHORS:
        w = raw[a - CONTEXT:a + HORIZON]
        if w.size != CONTEXT + HORIZON or not np.isfinite(w).all():
            return False, f"training_window_invalid anchor={a}"
    for o in ORIGINS:
        w = raw[o - CONTEXT:o + HORIZON]
        if w.size != CONTEXT + HORIZON or not np.isfinite(w).all():
            return False, f"eval_window_invalid origin={o}"
    return True, ""


def _candidate() -> Candidate:
    program = Program.from_steps([("outlier_mad", {})], source="grid0_census")
    return Candidate(
        candidate_id="grid0_census_probe",
        kind=CandidateKind.PROGRAM,
        program=program,
        source="grid0_census",
    )


def _window_checks(raw: np.ndarray) -> dict[str, Any]:
    """逐钉住训练窗口跑 verifier 原语；不评估 Consumer、不计算 gain。"""
    cand = _candidate()
    per_window = []
    passed_all = True
    total_modified_points = 0
    for a in PINNED_ANCHORS:
        w = np.asarray(raw[a - CONTEXT:a + HORIZON], dtype=np.float64)
        art = verify_candidate(
            cand, w,
            allowed_operators=("outlier_mad",),
            inspected_regions=((0, int(w.size)),),
            maximum_modified_fraction=MAX_MODIFIED_FRACTION,
            preserve_outside_inspected_region=True,
            require_finite_output=False,
        )
        modified_points = len(art.modified_indices)
        total_modified_points += modified_points
        row = {
            "anchor": a,
            "selectable": art.selectable,
            "status": art.receipt.status,
            "rejection_code": art.receipt.rejection_code,
            "modified_fraction": art.receipt.modified_fraction,
            "modified_points": modified_points,
        }
        per_window.append(row)
        if not art.selectable:
            passed_all = False
    return {
        "verifier_passed": passed_all,
        "total_modified_points": total_modified_points,
        "acting": total_modified_points > 0,
        "per_window": per_window,
    }


def _load_kdd_series() -> list[dict[str, Any]]:
    z = np.load(PROJECT_ROOT / "data" / "kdd2018" / "series_cache.npz",
                allow_pickle=True)
    names = [str(x) for x in z["names"]]
    out = []
    for name, raw, length in zip(names, z["values"], z["lengths"]):
        out.append({
            "series_uid": name,
            "entity_id": name,
            "dataset_id": "kdd2018",
            "length": int(length),
            "raw": np.asarray(raw, dtype=np.float64),
            "frequency": "hourly",
            "natural_missing_count": 0,
        })
    return out


def _load_registry_series() -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    # 复用 guidance runner 的 registry / clean_base 索引；只加载 cohort_B
    # 候选数据集的 npy，避免无谓加载全 clean_base（WSL 挂载上很慢）。
    import run_v1_guidance_evolution as G  # noqa: PLC0415
    registry = G._n4_load_registry()
    clean = G._n4_clean_records()
    want = set(COHORT_B_ORDER)
    out = []
    summary = {ds: {"registry_total": 0, "clean_joined": 0, "missing_clean": 0}
               for ds in COHORT_B_ORDER}
    for r in registry:
        ds = str(r.get("dataset_id"))
        if ds not in want:
            continue
        summary[ds]["registry_total"] += 1
        uid = str(r.get("series_uid"))
        ent = clean.get(uid)
        if ent is None:
            summary[ds]["missing_clean"] += 1
            continue
        summary[ds]["clean_joined"] += 1
        rec = ent["record"]
        out.append({
            "series_uid": uid,
            "entity_id": str(r.get("entity_id")),
            "dataset_id": ds,
            "length": int(rec.get("length") or 0),
            "raw": np.load(ent["values"]).astype(np.float64),
            "frequency": str(rec.get("frequency") or ""),
            "natural_missing_count": int(rec.get("natural_missing_count") or 0),
            "exposure_class": str(r.get("exposure_class") or ""),
            "timestamps_sha": str(r.get("timestamps_sha") or ""),
        })
    return out, summary


def main() -> int:
    report: dict[str, Any] = {}
    if CHECKPOINT_REL.exists():
        try:
            report = json.loads(CHECKPOINT_REL.read_text(encoding="utf-8"))
        except Exception:
            report = {}

    # ---------------- cohort_A ----------------
    kdd = _load_kdd_series()
    a_reject: dict[str, list[str]] = {}
    a_reject_details: list[dict[str, Any]] = []
    a_qualified: list[dict[str, Any]] = []
    a_excluded_reasons: list[dict[str, Any]] = []
    for i, s in enumerate(kdd):
        name = s["entity_id"]
        if i % 25 == 0:
            print(f"cohort_A census {i}/{len(kdd)}", flush=True)
        if name in COHORT_A_EXCLUSIONS:
            a_excluded_reasons.append({"series": name, "reason": "excluded_by_protocol_list"})
            continue
        if s["frequency"] != "hourly" or s["length"] < 1008:
            a_reject.setdefault("C2_frequency_or_length", []).append(name)
            a_reject_details.append({"series": name, "reason": "C2_frequency_or_length"})
            continue
        ok, reason = _windows_ok(s["raw"])
        if not ok:
            a_reject.setdefault(f"C3_{reason}", []).append(name)
            a_reject_details.append({"series": name, "reason": f"C3_{reason}"})
            continue
        checks = _window_checks(s["raw"])
        if not checks["verifier_passed"]:
            for w in checks["per_window"]:
                if not w["selectable"]:
                    key = f"C5_verify_reject_{w['rejection_code'] or 'unknown'}"
                    a_reject.setdefault(key, []).append(name)
                    a_reject_details.append({
                        "series": name, "reason": key,
                        "rejection_code": w["rejection_code"],
                        "anchor": w["anchor"],
                        "modified_fraction": w["modified_fraction"]})
                    break
            continue
        if not checks["acting"]:
            a_reject.setdefault("C4_not_acting", []).append(name)
            a_reject_details.append({"series": name, "reason": "C4_not_acting",
                                     "total_modified_points": 0})
            continue
        a_qualified.append({"series": name, "length": s["length"],
                            "total_modified_points": checks["total_modified_points"],
                            "max_modified_fraction": max(
                                w["modified_fraction"] for w in checks["per_window"]),
                            "per_window": checks["per_window"]})
    a_qualified.sort(key=lambda x: natural_key(x["series"]))
    a_selected = a_qualified[:COHORT_A_N]

    # ---------------- cohort_B ----------------
    reg, reg_summary = _load_registry_series()
    b_result: dict[str, Any] = {}
    b_selected: list[dict[str, Any]] = []
    for ds in COHORT_B_ORDER:
        pool = [s for s in reg if s["dataset_id"] == ds]
        print(f"cohort_B {ds} pool={len(pool)}", flush=True)
        timestamps = {s["timestamps_sha"] for s in pool}
        b_reject: dict[str, list[str]] = {}
        b_reject_details: list[dict[str, Any]] = []
        qualified = []
        for s in pool:
            if s["exposure_class"] != "certified_virgin":
                b_reject.setdefault("C1_not_certified_virgin", []).append(s["entity_id"])
                b_reject_details.append({"series": s["entity_id"], "reason": "C1_not_certified_virgin"})
                continue
            if s["frequency"] != "hourly" or s["length"] < 1008:
                b_reject.setdefault("C2_frequency_or_length", []).append(s["entity_id"])
                b_reject_details.append({"series": s["entity_id"], "reason": "C2_frequency_or_length"})
                continue
            if s["natural_missing_count"] != 0:
                b_reject.setdefault("C3_natural_missing", []).append(s["entity_id"])
                b_reject_details.append({"series": s["entity_id"], "reason": "C3_natural_missing"})
                continue
            ok, reason = _windows_ok(s["raw"])
            if not ok:
                b_reject.setdefault(f"C3_{reason}", []).append(s["entity_id"])
                b_reject_details.append({"series": s["entity_id"], "reason": f"C3_{reason}"})
                continue
            checks = _window_checks(s["raw"])
            if not checks["verifier_passed"]:
                for w in checks["per_window"]:
                    if not w["selectable"]:
                        key = f"C5_verify_reject_{w['rejection_code'] or 'unknown'}"
                        b_reject.setdefault(key, []).append(s["entity_id"])
                        b_reject_details.append({
                            "series": s["entity_id"], "reason": key,
                            "rejection_code": w["rejection_code"],
                            "anchor": w["anchor"],
                            "modified_fraction": w["modified_fraction"]})
                        break
                continue
            if not checks["acting"]:
                b_reject.setdefault("C4_not_acting", []).append(s["entity_id"])
                b_reject_details.append({"series": s["entity_id"], "reason": "C4_not_acting",
                                         "total_modified_points": 0})
                continue
            qualified.append({"series": s["entity_id"],
                              "series_uid": s["series_uid"],
                              "length": s["length"],
                              "total_modified_points": checks["total_modified_points"],
                              "max_modified_fraction": max(
                                  w["modified_fraction"] for w in checks["per_window"]),
                              "per_window": checks["per_window"]})
        qualified.sort(key=lambda x: natural_key(x["series"]))
        chosen = qualified[:COHORT_B_N]
        b_result[ds] = {
            "registry_total": reg_summary[ds]["registry_total"],
            "clean_joined": reg_summary[ds]["clean_joined"],
            "missing_clean": reg_summary[ds]["missing_clean"],
            "registry_pool": len(pool),
            "timestamps_sha_unique": len(timestamps),
            "timestamps_sha": sorted(timestamps)[:1],
            "qualified": len(qualified),
            "selected": [x["series"] for x in chosen],
            "reject_counts": {k: len(v) for k, v in b_reject.items()},
            "reject_sample": {k: v[:20] for k, v in b_reject.items()},
            "reject_details": b_reject_details,
            "selected_detail": chosen,
        }
        if len(chosen) >= COHORT_B_N:
            b_selected = chosen
            b_result["dataset_selected"] = ds
            b_result["triggered_fallback"] = (ds != COHORT_B_ORDER[0])
            break
    if not b_selected:
        b_result["dataset_selected"] = None
        b_result["error"] = "no cohort_B dataset reached required qualified count"

    # ---------------- checkpoint ----------------
    report["census"] = {
        "step": "grid0 step 5 eligibility census",
        "generated_by": "evaluation/functional/run_grid0_census.py",
        "cohort_A": {
            "dataset": "kdd2018",
            "pool_size": len(kdd),
            "qualified": len(a_qualified),
            "selected": [x["series"] for x in a_selected],
            "selected_detail": a_selected,
            "reject_counts": {k: len(v) for k, v in a_reject.items()},
            "reject_sample": {k: v[:20] for k, v in a_reject.items()},
            "reject_details": a_reject_details,
            "excluded_by_protocol_list": [x["series"] for x in a_excluded_reasons],
        },
        "cohort_B": b_result,
        "implementation_note": (
            "ScopeExecutor.verify() 当前仅返回 pass/rejected_windows，"
            "不暴露逐窗口 modified_fraction 与修改点数；本 census 使用其内部"
            "同一 verifier 原语 verify_candidate() 读取这两个量。全程未调用"
            "evaluate()，未计算任何 gain。"
        ),
    }
    CHECKPOINT_REL.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_REL.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          default=str) + "\n", encoding="utf-8")
    summary_print = {
        "cohort_A": {"pool": len(kdd), "qualified": len(a_qualified),
                     "selected": len(a_selected),
                     "rejects": {k: len(v) for k, v in a_reject.items()}},
        "cohort_B": {},
        "cohort_B_dataset_selected": b_result.get("dataset_selected"),
        "cohort_B_fallback": b_result.get("triggered_fallback"),
    }
    for ds in COHORT_B_ORDER:
        if ds in b_result:
            summary_print["cohort_B"][ds] = {k: v for k, v in b_result[ds].items()
                                             if k not in ("reject_sample",)}
    print(json.dumps(summary_print, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
