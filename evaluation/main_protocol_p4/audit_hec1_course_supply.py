"""D2: HEC-1 course-supply scan. Inventory only; freezes nothing.

0 LLM calls, 0 Consumer fits, outcome_values_read = 0.  Reads whether
missing-aware sMASE is defined and deployment-visible features on the
pre-origin 192-point window.  Does not read gain / error / utility.
Does not touch readable[80:120] x held-out origins.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from evaluation.main_protocol_p4 import audit_candidate_cohort as cohorts
from evaluation.main_protocol_p4 import audit_main_experiment_supply as supply
from evaluation.main_protocol_p4 import natural_structure_features as x1
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from SelfEvolvingHarnessTS.contracts.observables import (
    OBSERVABLE_FEATURES,
    OBSERVABLE_NUMERIC_BIN_LABELS,
    observable_numeric_bin,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPLY_PATH = PROJECT_ROOT / "artifacts/main_protocol/p4s_main_experiment_supply.json"
LEDGER_PATH = PROJECT_ROOT / "artifacts/main_protocol/p4t_exposure_ledger.json"
CONTRACT_PATH = PROJECT_ROOT / "artifacts/main_protocol/p4u_main_experiment_contract.json"
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/p4ac_hec1_course_supply.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/p4ac_hec1_course_supply.md"

CONTEXT, HORIZON = preflight.CONTEXT, preflight.HORIZON
PERIOD = preflight.PERIOD
DATA_VERSION = preflight.DATA_VERSION
TASK_KIND = "forecast"

ORIGINS = tuple(sorted(set(supply.READ_ORIGINS) | set(supply.CANDIDATE_ORIGINS)))
HELD_OUT = (4056, 4296, 4536, 4776, 5016)
HELD_IN = (1896, 2136, 2376, 2616, 2856)
SOURCE_READ = (1896, 2136, 2376, 2616, 2856)
CONSERVATIVE_MAX = 3816
OFFSETS = (0, 48, 144, 240)

BLOCK_SPANS = (
    (0, 40),
    (40, 80),
    (80, 120),
    (120, 160),
    (160, 200),
    (200, 239),
)
PHASE_T_BLOCKS = ((0, 40), (40, 80), (80, 120), (120, 160))
PHASE_S_BLOCKS = ((160, 200), (200, 239))

NUMERIC_FEATURES = tuple(
    name for name, kind in OBSERVABLE_FEATURES.items() if kind == "number"
)
BIN_INDEX = {label: i for i, label in enumerate(OBSERVABLE_NUMERIC_BIN_LABELS)}
LEVEL_OFFSET_FIRST_EDGE = 0.0  # default edges for estimated_level_offset start at 0


def _slice_name(span: tuple[int, int]) -> str:
    return "[%d:%d]" % span


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("%s is not an object" % path)
    return payload


def _faces(uids: list[str]) -> dict[str, Any]:
    face_a = list(uids[:20])
    face_b = list(uids[20:])
    return {
        "n": len(uids),
        "face_a": face_a,
        "face_b": face_b,
        "n_face_a": len(face_a),
        "n_face_b": len(face_b),
        "faces_equal_length": len(face_a) == len(face_b) and len(face_a) > 0,
    }


def _held_out_pair(block: tuple[int, int], origin: int) -> bool:
    return block == (80, 120) and int(origin) in HELD_OUT


def _degenerate_serving(raw: np.ndarray, origin: int) -> str | None:
    if raw.size < origin:
        return "series shorter than origin"
    window = np.asarray(raw[origin - CONTEXT:origin], dtype=np.float64)
    if window.size != CONTEXT:
        return "serving context shorter than 192"
    try:
        completed = forecast_runtime._linear_integrity(window)
        _c, _s, method = forecast_runtime._center_scale(np, completed)
    except Exception as exc:  # noqa: BLE001 - a 0-fit legality gate
        return "%s: %s" % (type(exc).__name__, str(exc)[:80])
    if method == "scale_floor_fallback":
        return "scale_floor_fallback"
    return None


def _evaluability_with_context(
    variant: dict[str, np.ndarray], uids: list[str], origin: int,
) -> dict[str, Any]:
    rows = cohorts.evaluability(variant, uids, [int(origin)])
    row = dict(rows[0])
    degenerate = []
    for uid in uids:
        why = _degenerate_serving(variant[uid], int(origin))
        if why:
            degenerate.append({"uid": uid, "why": why})
    row["serving_context_degenerate"] = degenerate
    if degenerate:
        row["usable"] = False
        extra = list(row.get("not_evaluable_series") or [])
        extra.extend(degenerate)
        row["not_evaluable_series"] = extra
        if not row.get("reason"):
            row["reason"] = "raw serving context degenerate"
    return row


def _exposure_labels(block: tuple[int, int], origin: int) -> list[str]:
    labels: list[str] = []
    o = int(origin)
    if _held_out_pair(block, o):
        return ["HELD_OUT_FROZEN"]
    if block in ((0, 40), (40, 80)) and o in supply.READ_ORIGINS:
        labels.append("SPENT_DEV")
    if block == (160, 200):
        source_windows = set(SOURCE_READ)
        for base in SOURCE_READ:
            source_windows.add(base + 48)
            source_windows.add(base + 240)
        if o in source_windows:
            labels.append("SOURCE_V1–V3_READ")
    if block == (80, 120) and o in HELD_IN:
        labels.append("TARGET_HELD_IN")
    if not labels:
        labels.append("UNREAD")
    return labels


def _public_card(window: np.ndarray) -> dict[str, Any]:
    card = dict(extract_public_features(window, task_kind=TASK_KIND))
    structure = x1.extract(window, period=PERIOD)
    card.update(structure)
    return card


def _bin_vector(card: dict[str, Any]) -> tuple[str, ...]:
    labels = []
    for name in NUMERIC_FEATURES:
        value = card.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            labels.append("missing")
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            labels.append("nonfinite")
            continue
        try:
            labels.append(observable_numeric_bin(name, numeric))
        except ValueError:
            labels.append("unbinned")
    return tuple(labels)


def _level_offset_material(card: dict[str, Any]) -> bool:
    value = card.get("estimated_level_offset")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return abs(float(value)) > LEVEL_OFFSET_FIRST_EDGE


def _prevalence_row(
    variant: dict[str, np.ndarray], face_a: list[str], origin: int,
) -> dict[str, Any]:
    z_members: list[str] = []
    missing_gt_0 = 0
    missing_fracs: list[float] = []
    gap_runs: list[int] = []
    level_material = 0
    bins: list[tuple[str, ...]] = []
    for uid in face_a:
        raw = np.asarray(variant[uid], dtype=np.float64)
        window = raw[origin - CONTEXT:origin]
        card = _public_card(window)
        z_peak = float(card.get("local_robust_z_peak") or 0.0)
        if z_peak >= 3.0:
            z_members.append(uid)
        miss = float(card.get("missing_fraction") or 0.0)
        missing_fracs.append(miss)
        if miss > 0.0:
            missing_gt_0 += 1
        gaps = ~np.isfinite(window)
        gap_runs.append(cohorts.longest_run(gaps))
        if _level_offset_material(card):
            level_material += 1
        bins.append(_bin_vector(card))
    return {
        "n_z_peak_ge_3": len(z_members),
        "z_peak_ge_3_members": z_members,
        "n_missing_gt_0": missing_gt_0,
        "missing_fraction_median": (
            float(np.median(missing_fracs)) if missing_fracs else None),
        "missing_fraction_max": (
            float(np.max(missing_fracs)) if missing_fracs else None),
        "longest_gap_length_median": (
            float(np.median(gap_runs)) if gap_runs else None),
        "n_level_offset_material": level_material,
        "n_unique_binned_vectors": len(set(bins)),
        "binned_vector_dim": len(NUMERIC_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
    }


def _jaccard(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _phase_t_origins(block: tuple[int, int], usable: dict[tuple, list[int]],
                     cap: int | None) -> list[int]:
    origins = list(usable.get(block, ()))
    if cap is not None:
        origins = [o for o in origins if o <= cap]
    if block == (80, 120):
        held = [o for o in HELD_IN if o in set(usable.get(block, ()))]
        extra = [o for o in origins if o <= CONSERVATIVE_MAX]
        origins = sorted(set(held) | set(extra))
        if cap is not None:
            origins = [o for o in origins if o <= cap]
    return origins


def _units(blocks: tuple[tuple[int, int], ...], usable: dict[tuple, list[int]],
           cap: int | None, extra_held_in: bool) -> list[dict[str, Any]]:
    rows = []
    for block in blocks:
        origins = list(usable.get(block, ()))
        if cap is not None:
            origins = [o for o in origins if o <= cap]
        if extra_held_in and block == (80, 120):
            origins = _phase_t_origins(block, usable, cap)
        for origin in origins:
            rows.append({
                "block": _slice_name(block),
                "span": list(block),
                "origin": int(origin),
            })
    return rows


def _orderings(units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    block_order = [_slice_name(span) for span in PHASE_T_BLOCKS]
    for row in units:
        by_block.setdefault(row["block"], []).append(row)
    for name in by_block:
        by_block[name] = sorted(by_block[name], key=lambda r: r["origin"])
    forward = []
    for name in block_order:
        forward.extend(by_block.get(name, ()))
    reverse = list(reversed(forward))
    interleaved = []
    pointers = {name: 0 for name in block_order}
    progressed = True
    while progressed:
        progressed = False
        for name in block_order:
            rows = by_block.get(name, ())
            idx = pointers[name]
            if idx < len(rows):
                interleaved.append(rows[idx])
                pointers[name] = idx + 1
                progressed = True
    return {
        "forward": forward,
        "reverse": reverse,
        "interleaved": interleaved,
        "rules": {
            "forward": (
                "block order [0:40]→[40:80]→[80:120]→[120:160], "
                "origin ascending within block"
            ),
            "reverse": "reverse of forward",
            "interleaved": (
                "round-robin across the four Phase-T blocks, "
                "origin ascending within block"
            ),
        },
    }


def _cut_conclusion(last: dict[str, Any]) -> dict[str, Any]:
    leftover = last["face_a"][19:] + last["face_b"][19:] if last["n"] == 39 else []
    equal = {
        "face_a": last["uids"][:19],
        "face_b": last["uids"][19:38],
        "leftover": last["uids"][38:],
        "n_face_a": 19,
        "n_face_b": 19,
    } if last["n"] == 39 else None
    return {
        "n_series": last["n"],
        "canonical_cut": {
            "face_a_n": last["n_face_a"],
            "face_b_n": last["n_face_b"],
            "faces_equal_length": last["faces_equal_length"],
            "rule": "face A = first 20, face B = remainder (19 when n=39)",
        },
        "equal_face_cut": equal,
        "pipeline_20_20_formable": last["n"] >= 40,
        "conclusion": (
            "[200:239] has 39 series so a 20/20 cell does not form. "
            "Canonical cut for this scan is A=20 / B=19 as the task book "
            "states. If a later freeze requires equal faces, the alternative "
            "is A=19 / B=19 with leftover %s excluded; this scan does not "
            "freeze either cut." % (equal["leftover"] if equal else [])
        ),
        "leftover_if_equal_cut": leftover if last["n"] == 39 else [],
    }


def build() -> dict[str, Any]:
    supply_art = _load_json(SUPPLY_PATH)
    ledger = _load_json(LEDGER_PATH)
    contract = _load_json(CONTRACT_PATH)
    readable = list(supply_art["readable_uids"])
    if len(readable) != 239:
        raise RuntimeError("p4s readable_uids length is %d, expected 239"
                           % len(readable))
    why = supply_art["boundary_caveat_for_the_ruling"][
        "why_it_is_not_outcome_selection"]
    variant = preflight.load_variant()

    blocks = []
    block_uids: dict[tuple[int, int], list[str]] = {}
    for span in BLOCK_SPANS:
        uids = list(readable[span[0]:span[1]])
        faces = _faces(uids)
        block_uids[span] = uids
        blocks.append({
            "slice": _slice_name(span),
            "span": list(span),
            "uids": uids,
            **faces,
        })

    usability: list[dict[str, Any]] = []
    usable_by_block: dict[tuple[int, int], list[int]] = {span: [] for span in BLOCK_SPANS}
    for span in BLOCK_SPANS:
        uids = block_uids[span]
        for origin in ORIGINS:
            labels = _exposure_labels(span, origin)
            if _held_out_pair(span, origin):
                usability.append({
                    "block": _slice_name(span),
                    "span": list(span),
                    "origin": int(origin),
                    "usable": None,
                    "touched": False,
                    "exposure": labels,
                    "reason": "HELD_OUT_FROZEN named only; not read",
                })
                continue
            row = _evaluability_with_context(variant, uids, origin)
            usable = bool(row.get("usable"))
            if usable:
                usable_by_block[span].append(int(origin))
            usability.append({
                "block": _slice_name(span),
                "span": list(span),
                "origin": int(origin),
                "usable": usable,
                "touched": True,
                "exposure": labels,
                "min_observed_truth": row.get("min_observed_truth"),
                "not_evaluable_n": len(row.get("not_evaluable_series") or []),
                "serving_context_degenerate_n": len(
                    row.get("serving_context_degenerate") or []),
                "reason": row.get("reason"),
            })

    prevalence: list[dict[str, Any]] = []
    members_by_unit: dict[tuple[str, int], set[str]] = {}
    for span in BLOCK_SPANS:
        face_a = block_uids[span][:20]
        for origin in usable_by_block[span]:
            stats = _prevalence_row(variant, face_a, origin)
            members_by_unit[(_slice_name(span), int(origin))] = set(
                stats.pop("z_peak_ge_3_members"))
            prevalence.append({
                "block": _slice_name(span),
                "origin": int(origin),
                "face_a_n": len(face_a),
                **stats,
                "jaccard_with_next_usable_origin": None,
                "next_origin": None,
            })

    by_block_rows: dict[str, list[dict[str, Any]]] = {}
    for row in prevalence:
        by_block_rows.setdefault(row["block"], []).append(row)
    jaccards: list[float] = []
    for name, rows in by_block_rows.items():
        rows.sort(key=lambda r: r["origin"])
        for left, right in zip(rows, rows[1:]):
            jac = _jaccard(
                members_by_unit[(name, left["origin"])],
                members_by_unit[(name, right["origin"])],
            )
            left["jaccard_with_next_usable_origin"] = jac
            left["next_origin"] = right["origin"]
            if jac is not None:
                jaccards.append(float(jac))

    last_block = next(b for b in blocks if b["slice"] == "[200:239]")
    cut = _cut_conclusion(last_block)

    phase_s_all = _units(PHASE_S_BLOCKS, usable_by_block, None, False)
    phase_s_le = _units(PHASE_S_BLOCKS, usable_by_block, CONSERVATIVE_MAX, False)
    for row in phase_s_all + phase_s_le:
        span = (int(row["span"][0]), int(row["span"][1]))
        row["exposure"] = _exposure_labels(span, row["origin"])
        row["already_read"] = "SOURCE_V1–V3_READ" in row["exposure"]

    phase_t_all = _units(PHASE_T_BLOCKS, usable_by_block, None, True)
    phase_t_le = _units(PHASE_T_BLOCKS, usable_by_block, CONSERVATIVE_MAX, True)
    for row in phase_t_all + phase_t_le:
        span = (int(row["span"][0]), int(row["span"][1]))
        row["exposure"] = _exposure_labels(span, row["origin"])

    orderings = _orderings(phase_t_le)

    unique_bins = [
        int(row["n_unique_binned_vectors"]) for row in prevalence]
    sparse = [
        row for row in prevalence if int(row["n_z_peak_ge_3"]) < 5]
    composition = {
        "repeat_pattern_family": {
            "what": "same-block adjacent-origin Jaccard of z_peak>=3 members",
            "jaccard": _quantiles(jaccards),
            "empty": not jaccards,
        },
        "within_family_heterogeneity": {
            "what": "unique 12-d binned public-feature vectors on face A",
            "n_unique_binned_vectors": _quantiles(
                [float(v) for v in unique_bins]),
            "empty": not unique_bins,
            "note": (
                "task book says 22-d; this checkout's numeric observable "
                "vocabulary is %d names (public card is 21 keys + 6 X1 "
                "structure descriptors, X1 has no frozen numeric bins)"
                % len(NUMERIC_FEATURES)
            ),
        },
        "pattern_sparse_units": {
            "rule": "n_z_peak_ge_3 < 5",
            "n": len(sparse),
            "units": [
                {"block": r["block"], "origin": r["origin"],
                 "n_z_peak_ge_3": r["n_z_peak_ge_3"]}
                for r in sparse
            ],
            "empty": not sparse,
        },
    }

    held_out_series = list(ledger.get("target_series") or [])
    held_out_origins = list(ledger.get("proposed_held_out") or list(HELD_OUT))
    held_out_pairs = {
        (str(uid), int(origin))
        for uid in held_out_series
        for origin in held_out_origins
    }
    candidate_units = phase_s_all + phase_t_all
    window_labels = []
    intersection = []
    plus144_vs_heldout = []
    seen_windows = set()
    for unit in candidate_units:
        span = (int(unit["span"][0]), int(unit["span"][1]))
        uids = block_uids[span]
        origin = int(unit["origin"])
        for offset in OFFSETS:
            window = origin + offset
            key = (unit["block"], origin, offset, window)
            if key in seen_windows:
                continue
            seen_windows.add(key)
            hits = sorted(
                uid for uid in uids if (uid, window) in held_out_pairs
            )
            label = {
                "block": unit["block"],
                "candidate_origin": origin,
                "offset": offset,
                "window": window,
                "exposure_at_candidate_origin": unit.get("exposure"),
                "held_out_series_at_window": hits,
            }
            window_labels.append(label)
            if hits:
                intersection.append(label)
            if offset == 144:
                plus144_vs_heldout.append({
                    "block": unit["block"],
                    "candidate_origin": origin,
                    "window": window,
                    "overlaps_held_out_origin": window in set(held_out_origins),
                    "held_out_series_at_window": hits,
                })

    counts = {}
    for span in BLOCK_SPANS:
        all_o = usable_by_block[span]
        counts[_slice_name(span)] = {
            "all": len(all_o),
            "le_3816": len([o for o in all_o if o <= CONSERVATIVE_MAX]),
            "origins_all": all_o,
            "origins_le_3816": [o for o in all_o if o <= CONSERVATIVE_MAX],
        }

    report = {
        "stage": "P4AC_HEC1_COURSE_SUPPLY",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_SUPPLY_SCAN",
        "data_version": DATA_VERSION,
        "sources": {
            "supply": str(SUPPLY_PATH.relative_to(PROJECT_ROOT).as_posix()),
            "exposure_ledger": str(LEDGER_PATH.relative_to(PROJECT_ROOT).as_posix()),
            "contract": str(CONTRACT_PATH.relative_to(PROJECT_ROOT).as_posix()),
        },
        "origin_grid": list(ORIGINS),
        "blocks": [
            {k: v for k, v in block.items() if k != "uids"} | {
                "face_a": block["face_a"],
                "face_b": block["face_b"],
            }
            for block in blocks
        ],
        "cut_200_239": cut,
        "usability": usability,
        "usability_counts": counts,
        "prevalence": prevalence,
        "proposals": {
            "phase_s": phase_s_all,
            "phase_s_le_3816": phase_s_le,
            "phase_t_all": phase_t_all,
            "phase_t_le_3816": phase_t_le,
            "orderings": orderings,
            "n_phase_s_all": len(phase_s_all),
            "n_phase_s_le_3816": len(phase_s_le),
            "n_phase_t_all": len(phase_t_all),
            "n_phase_t_le_3816": len(phase_t_le),
        },
        "composition_check": composition,
        "exposure_cross_check": {
            "held_out_intersection": intersection,
            "held_out_intersection_empty": not intersection,
            "per_window_labels": window_labels,
            "plus144_vs_held_out": plus144_vs_heldout,
            "held_out_pairs_named": len(held_out_pairs),
            "p4t_verdict": ledger.get("verdict"),
            "p4u_held_out": contract.get("geometry", {}).get("held_out_origins"),
        },
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": 0,
            "outcome_values_read": 0,
            "held_out_reads": 0,
            "thresholds_changed": 0,
            "anything_frozen_by_this_audit": False,
            "why_it_is_not_outcome_selection": why,
        },
        "deviations": [
            {
                "what": "22-d binned vector",
                "why": (
                    "observable numeric vocabulary in this checkout is %d "
                    "features; public card has 21 keys. Heterogeneity proxy "
                    "uses the %d numeric observables' frozen bins."
                    % (len(NUMERIC_FEATURES), len(NUMERIC_FEATURES))
                ),
            },
            {
                "what": "raw serving-context non-degeneracy",
                "why": (
                    "evaluability() does not check it; this audit adds a 0-fit "
                    "_linear_integrity + _center_scale gate and fails usable "
                    "when method == scale_floor_fallback"
                ),
            },
        ],
        "spec_tensions": [
            {
                "what": "[200:239] vs p4s [200:240] 'no cohort forms'",
                "reading": cut["conclusion"],
            },
        ],
        "releases": "NONE",
    }
    return report


def _md(report: dict[str, Any]) -> str:
    counts = report["usability_counts"]
    lines = [
        "# p4ac HEC-1 course supply scan",
        "",
        "0 LLM / 0 Consumer fit / outcome_values_read 0. Inventory only.",
        "",
        "## (1) Six blocks × origin, two calibers",
        "",
        "| block | all usable | ≤3816 | origins ≤3816 |",
        "| --- | ---: | ---: | --- |",
    ]
    for span in BLOCK_SPANS:
        name = _slice_name(span)
        row = counts[name]
        lines.append("| %s | %d | %d | %s |" % (
            name, row["all"], row["le_3816"],
            ", ".join(str(o) for o in row["origins_le_3816"]) or "—"))
    cut = report["cut_200_239"]
    prop = report["proposals"]
    comp = report["composition_check"]
    xcheck = report["exposure_cross_check"]
    lines += [
        "",
        "## (2) [200:239] cut",
        "",
        cut["conclusion"],
        "",
        "## (3) Phase S / T unit counts",
        "",
        "- Phase S all / ≤3816: **%d** / **%d**" % (
            prop["n_phase_s_all"], prop["n_phase_s_le_3816"]),
        "- Phase T all / ≤3816: **%d** / **%d**" % (
            prop["n_phase_t_all"], prop["n_phase_t_le_3816"]),
        "",
        "## (4) Composition three-element check",
        "",
        "- Repeat-family Jaccard empty=%s n=%s median=%s" % (
            comp["repeat_pattern_family"]["empty"],
            comp["repeat_pattern_family"]["jaccard"]["n"],
            comp["repeat_pattern_family"]["jaccard"]["median"]),
        "- Heterogeneity unique-bins empty=%s median=%s" % (
            comp["within_family_heterogeneity"]["empty"],
            comp["within_family_heterogeneity"]["n_unique_binned_vectors"]["median"]),
        "- Sparse units (n_z_peak_ge_3<5) empty=%s n=%d" % (
            comp["pattern_sparse_units"]["empty"],
            comp["pattern_sparse_units"]["n"]),
        "",
        "## (5) Exposure intersection",
        "",
        "held-out intersection empty: **%s** (n=%d). p4t verdict: %s." % (
            xcheck["held_out_intersection_empty"],
            len(xcheck["held_out_intersection"]),
            xcheck.get("p4t_verdict")),
        "",
        "Per-window labels (+0/+48/+144/+240) live in the JSON "
        "`exposure_cross_check.per_window_labels`. `[80:120] × 2856 +144 = 3000` "
        "does not overlap any held-out origin.",
        "",
        "This scan freezes nothing.",
        "",
        "## (6) Deviations",
        "",
    ]
    for row in report.get("deviations") or ():
        lines.append("- %s: %s" % (row["what"], row["why"]))
    lines += ["", "## (7) Spec tensions", ""]
    for row in report.get("spec_tensions") or ():
        lines.append("- %s — %s" % (row["what"], row["reading"]))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    report = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(_md(report), encoding="utf-8")
    prop = report["proposals"]
    print("blocks                 : %d" % len(report["blocks"]))
    print("usable cells           : %d" % sum(
        1 for row in report["usability"] if row.get("usable") is True))
    print("phase S all / le_3816  : %d / %d" % (
        prop["n_phase_s_all"], prop["n_phase_s_le_3816"]))
    print("phase T all / le_3816  : %d / %d" % (
        prop["n_phase_t_all"], prop["n_phase_t_le_3816"]))
    print("held-out intersection  : %d" % len(
        report["exposure_cross_check"]["held_out_intersection"]))
    print("consumer_fits          : %s" % report["boundary"]["consumer_fits"])
    print("wrote %s" % OUT_JSON.relative_to(PROJECT_ROOT).as_posix())
    print("wrote %s" % OUT_MD.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
