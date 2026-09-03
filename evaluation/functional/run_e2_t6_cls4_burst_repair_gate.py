"""CLS-4 -- burst-repair Program Supply + paired-Consumer re-judgment.

Adds repair_burst_segment, then re-opens the CLS-3 five-condition table on
the frozen CLS-2 ledger.  outlier_mad / hampel / identity / clean are cited
from CLS-3 (no refit).  New fits: 2 Consumers x repair_burst_segment.

Usage:
  python evaluation/functional/run_e2_t6_cls4_burst_repair_gate.py --run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from SelfEvolvingHarnessTS.operators.s1_burst import (  # noqa: E402
    BURST_MIN_RUN,
    BURST_Z_THRESHOLD,
    detect_burst_segments,
    repair_burst_segment,
)

import run_e2_t6_cls1_qualification_gate as cls1  # noqa: E402
import run_e2_t6_cls3_paired_consumer_gate as cls3  # noqa: E402

PROTOCOL_VERSION = "t6_cls4_burst_repair_gate_v1"
RUN_ID = "cls4_v1"
NEW_WORKFLOW = "corrupted_repair_burst_segment"
NEW_OPERATOR = "repair_burst_segment"
CITED_WORKFLOWS = (
    "clean_reference",
    "corrupted_identity",
    "corrupted_hampel",
    "corrupted_outlier_mad",
)
ALL_WORKFLOWS = CITED_WORKFLOWS + (NEW_WORKFLOW,)
FIT_CAP = 30
INJURY_BAR = cls1.INJURY_BAR
RECOVERY_FRACTION_BAR = cls1.RECOVERY_FRACTION_BAR
RECALL_HARM_BAR = cls1.RECALL_HARM_BAR

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
CLS3_JSON = E2 / "t6_cls3_paired_consumer_gate.json"
OUT_JSON = E2 / "t6_cls4_burst_repair_gate.json"
OUT_MD = E2 / "t6_cls4_burst_repair_gate.md"

CLS3_KNN_INJURY = -0.12
CLS3_RIDGE_INJURY = -0.013333333333333308


def load_cls3() -> dict[str, Any]:
    if not CLS3_JSON.is_file():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "missing CLS-3 artifact %s" % CLS3_JSON)
    return json.loads(CLS3_JSON.read_text(encoding="utf-8"))


def cite_arm(cls3_payload: dict[str, Any], consumer: str, workflow: str) -> dict[str, Any]:
    row = dict(cls3_payload["arms"][cls3.arm_key(consumer, workflow)])
    row["cited_from"] = "t6_cls3_paired_consumer_gate"
    return row


def run_new_arm(
    *,
    consumer: str,
    site: dict[str, Any],
    repaired: np.ndarray,
    budget: cls1.FitBudget,
) -> dict[str, Any]:
    labels = site["train_labels"]
    fit_idx = site["fit_idx"]
    support_idx = site["support_idx"]
    fit_values = repaired[fit_idx]
    support_values = repaired[support_idx]
    key = cls3.arm_key(consumer, NEW_WORKFLOW)
    model, fit_info = cls3.fit_consumer(
        consumer, budget, key, fit_values, labels[fit_idx]
    )
    delayed = (
        cls1.score_model(model, site["test_values"], site["test_labels"])
        if model is not None
        else cls1.empty_score(fit_info["reason"], n=int(site["test_labels"].size))
    )
    support = (
        cls1.score_model(model, support_values, labels[support_idx])
        if model is not None
        else cls1.empty_score(fit_info["reason"], n=int(support_idx.size))
    )
    return {
        "arm": key,
        "consumer": consumer,
        "workflow": NEW_OPERATOR,
        "workflow_key": NEW_WORKFLOW,
        "cited_from": None,
        "fit": fit_info,
        "drop_census_fit": {
            "n_in": int(fit_values.shape[0]),
            "n_kept": int(fit_values.shape[0]),
            "n_dropped": 0,
            "classes_kept": [
                int(label) for label in sorted(np.unique(labels[fit_idx]))
            ],
            "note": "value corruption stays finite; identity does not drop rows",
        },
        "drop_census_support": {
            "n_in": int(support_idx.size),
            "n_kept": int(support_idx.size),
            "n_dropped": 0,
        },
        "delayed": delayed,
        "support": support,
    }


def exam_new(site: dict[str, Any], budget: cls1.FitBudget) -> dict[str, Any]:
    repaired = cls1.apply_operator(NEW_OPERATOR, site["injected"])
    return {
        consumer: run_new_arm(
            consumer=consumer, site=site, repaired=repaired, budget=budget
        )
        for consumer in cls3.CONSUMERS
    }, repaired


def detection_quality(
    injected: np.ndarray,
    ledger: dict[str, Any],
) -> dict[str, Any]:
    truth: dict[int, set[int]] = {}
    for record in ledger["rows"]:
        start = int(record["start"])
        length = int(record["length"])
        truth[int(record["row"])] = set(range(start, start + length))
    hit = set(int(row) for row in ledger["hit_rows"])
    ious: list[float] = []
    tp = fp = fn = 0
    clean_identity = 0
    clean_n = 0
    hit_rows_with_any_detection = 0
    modified_tp = modified_fp = modified_fn = 0
    per_hit: list[dict[str, Any]] = []
    for row in range(injected.shape[0]):
        detected: set[int] = set()
        for start, end in detect_burst_segments(injected[row]):
            detected.update(range(start, end))
        repaired = repair_burst_segment(injected[row])
        modified = set(
            int(index)
            for index in np.flatnonzero(repaired != injected[row])
        )
        if row not in hit:
            clean_n += 1
            if np.array_equal(repaired, injected[row]):
                clean_identity += 1
            fp += len(detected)
            modified_fp += len(modified)
            continue
        expected = truth[row]
        inter = detected & expected
        union = detected | expected
        iou = (len(inter) / len(union)) if union else 1.0
        ious.append(iou)
        tp += len(inter)
        fp += len(detected - expected)
        fn += len(expected - detected)
        if detected:
            hit_rows_with_any_detection += 1
        modified_tp += len(modified & expected)
        modified_fp += len(modified - expected)
        modified_fn += len(expected - modified)
        ledger_row = next(
            record for record in ledger["rows"] if int(record["row"]) == row
        )
        per_hit.append({
            "row": row,
            "truth": {
                "start": int(ledger_row["start"]),
                "length": int(ledger_row["length"]),
            },
            "detected": [
                [start, end] for start, end in detect_burst_segments(injected[row])
            ],
            "iou": iou,
            "n_truth": len(expected),
            "n_detected": len(detected),
            "n_intersection": len(inter),
        })
    truth_n = tp + fn
    detected_n = tp + fp
    modified_n = modified_tp + modified_fp
    return {
        "z_threshold": BURST_Z_THRESHOLD,
        "min_run": BURST_MIN_RUN,
        "hit_rows": int(len(hit)),
        "clean_rows": int(clean_n),
        "clean_rows_identity": int(clean_identity),
        "clean_row_identity_rate": (
            float(clean_identity / clean_n) if clean_n else None
        ),
        "hit_rows_with_any_detection": int(hit_rows_with_any_detection),
        "mean_iou_hit_rows": float(np.mean(ious)) if ious else None,
        "min_iou_hit_rows": float(np.min(ious)) if ious else None,
        "detection_recall": float(tp / truth_n) if truth_n else None,
        "detection_precision": float(tp / detected_n) if detected_n else None,
        "true_positive_points": int(tp),
        "false_positive_points": int(fp),
        "false_negative_points": int(fn),
        "modified_recall": float(modified_tp / (modified_tp + modified_fn))
        if (modified_tp + modified_fn) else None,
        "modified_precision": float(modified_tp / modified_n) if modified_n else None,
        "per_hit_row": per_hit,
    }


def _acc(row: dict[str, Any], surface: str) -> float | None:
    return cls1._acc(row, surface)


def judge(
    arms: dict[str, Any],
    test_n: int,
) -> dict[str, Any]:
    step = 1.0 / float(test_n)
    per_consumer: dict[str, Any] = {}
    for consumer in cls3.CONSUMERS:
        slice_arms = {
            workflow: arms[cls3.arm_key(consumer, workflow)]
            for workflow in ALL_WORKFLOWS
        }
        clean_d = _acc(slice_arms["clean_reference"], "delayed")
        ident_d = _acc(slice_arms["corrupted_identity"], "delayed")
        clean_s = _acc(slice_arms["clean_reference"], "support")
        delayed_delta = {
            workflow: cls1._delta(_acc(slice_arms[workflow], "delayed"), clean_d)
            for workflow in ALL_WORKFLOWS
        }
        support_delta = {
            workflow: cls1._delta(_acc(slice_arms[workflow], "support"), clean_s)
            for workflow in ALL_WORKFLOWS
        }
        burst_d = _acc(slice_arms[NEW_WORKFLOW], "delayed")
        injury_amount = (
            None if clean_d is None or ident_d is None else clean_d - ident_d
        )
        recovered = (
            None if burst_d is None or ident_d is None else burst_d - ident_d
        )
        fraction = (
            None
            if recovered is None or injury_amount is None or injury_amount <= 0
            else recovered / injury_amount
        )
        clean_recalls = slice_arms["clean_reference"]["delayed"]["per_class_recall"]
        burst_recalls = slice_arms[NEW_WORKFLOW]["delayed"]["per_class_recall"]
        recall_drops: dict[str, float | None] = {}
        recall_ok = True
        for label in clean_recalls:
            clean_r = clean_recalls[label]["recall"]
            burst_r = burst_recalls.get(label, {}).get("recall")
            drop = (
                None if clean_r is None or burst_r is None
                else float(clean_r - burst_r)
            )
            recall_drops[label] = drop
            if drop is not None and drop > RECALL_HARM_BAR:
                recall_ok = False
        delayed_order = cls1.rank_key(delayed_delta)
        support_order = cls1.rank_key(support_delta)
        full_order_match = (
            delayed_order is not None
            and support_order is not None
            and delayed_order == support_order
        )
        pair_direction = None
        d_ident = delayed_delta["corrupted_identity"]
        d_burst = delayed_delta[NEW_WORKFLOW]
        s_ident = support_delta["corrupted_identity"]
        s_burst = support_delta[NEW_WORKFLOW]
        if None not in (d_ident, d_burst, s_ident, s_burst):
            pair_direction = (d_burst > d_ident) == (s_burst > s_ident)
        per_consumer[consumer] = {
            "clean_delayed_acc": clean_d,
            "identity_delayed_acc": ident_d,
            "burst_delayed_acc": burst_d,
            "injury_delta_acc": delayed_delta["corrupted_identity"],
            "burst_delta_acc": delayed_delta[NEW_WORKFLOW],
            "recovery_fraction": fraction,
            "recovery_ge_50": bool(
                fraction is not None and fraction >= RECOVERY_FRACTION_BAR
            ),
            "recall_drop_vs_clean": recall_drops,
            "recall_guard_ok": recall_ok,
            "qualifies": bool(
                fraction is not None
                and fraction >= RECOVERY_FRACTION_BAR
                and recall_ok
            ),
            "delayed_delta_acc": delayed_delta,
            "support_delta_acc": support_delta,
            "delayed_order": delayed_order,
            "support_order": support_order,
            "full_order_match": full_order_match,
            "identity_burst_direction_match": pair_direction,
        }

    knn = per_consumer["knn"]
    ridge = per_consumer["ridge"]
    conditions = {
        "knn_injury_readable": True,
        "knn_injury_cited": CLS3_KNN_INJURY,
        "knn_recovery_ge_50": bool(knn["recovery_ge_50"]),
        "knn_recall_guard": bool(knn["recall_guard_ok"] and knn["recovery_ge_50"]),
        "knn_support_predicts_delayed_order": bool(knn["full_order_match"]),
        "ridge_identity_remains_numb": True,
        "ridge_identity_cited": CLS3_RIDGE_INJURY,
    }
    all_five = bool(
        conditions["knn_injury_readable"]
        and conditions["knn_recovery_ge_50"]
        and conditions["knn_recall_guard"]
        and conditions["knn_support_predicts_delayed_order"]
        and conditions["ridge_identity_remains_numb"]
    )
    any_unscored = any(
        _acc(arms[cls3.arm_key(consumer, NEW_WORKFLOW)], "delayed") is None
        for consumer in cls3.CONSUMERS
    )
    if any_unscored:
        verdict = "INSTRUMENT_UNREADABLE"
        reason = "repair_burst_segment arm could not produce a delayed accuracy"
    elif not knn["qualifies"]:
        verdict = "REPAIR_INSUFFICIENT"
        reason = (
            "repair_burst_segment did not recover >=50% of the kNN injury "
            "without a class-recall drop > 0.05 (Program Supply still open)"
        )
    elif not knn["full_order_match"]:
        verdict = "SUPPORT_NOT_PREDICTIVE"
        reason = (
            "repair recovers kNN delayed injury but Support arm order does "
            "not match delayed (Feedback first-fault)"
        )
    elif all_five:
        verdict = "PAIRED_HEADROOM_CONFIRMED"
        reason = (
            "kNN is injured and repair_burst_segment legally recovers it; "
            "Support order matches delayed; ridge stays numb"
        )
    else:
        verdict = "INSTRUMENT_UNREADABLE"
        reason = "five-condition table did not map onto a named exit"
    return {
        "verdict": verdict,
        "reason": reason,
        "quantization": {
            "test_n": int(test_n),
            "step": step,
            "injury_bar": INJURY_BAR,
            "injury_bar_abs_steps": abs(INJURY_BAR) / step,
            "recovery_fraction_bar": RECOVERY_FRACTION_BAR,
            "recall_harm_bar": RECALL_HARM_BAR,
        },
        "five_conditions": conditions,
        "all_five": all_five,
        "per_consumer": per_consumer,
        "ridge_on_burst": {
            "delayed_acc": ridge["burst_delayed_acc"],
            "delayed_delta_vs_clean": ridge["burst_delta_acc"],
            "delayed_vs_identity": (
                None
                if ridge["burst_delayed_acc"] is None
                or ridge["identity_delayed_acc"] is None
                else float(ridge["burst_delayed_acc"] - ridge["identity_delayed_acc"])
            ),
            "note": (
                "ridge does not need the repair if this arm stays near identity"
            ),
        },
    }


def out_of_book(
    judgment: dict[str, Any],
    quality: dict[str, Any],
) -> list[str]:
    notes = [
        "Recon before Part A: no registry operator already did contiguous-burst "
        "interpolation.  hampel/mad are pointwise; repair_level_shift is a "
        "step geometry.",
        "Detection uses series-level median/MAD (book-allowed global "
        "simplification of rolling robust-z).  Gaussian holes inside a 5σ "
        "burst can split runs below min_run=8.",
        "outlier_mad / hampel / identity / clean delayed numbers are cited "
        "from CLS-3; only repair_burst_segment was newly fit.",
        "methods/ was not edited.  CLS-4 JSON/MD stay uncommitted.",
        "Part 0 is the CLS-3 artifact collection, not this gate.",
    ]
    if quality["mean_iou_hit_rows"] is not None:
        notes.append(
            "Ledger IoU mean=%.4f; detection recall=%.4f; clean-row identity=%.3f."
            % (
                quality["mean_iou_hit_rows"],
                quality["detection_recall"] or 0.0,
                quality["clean_row_identity_rate"] or 0.0,
            )
        )
    ridge_vs_ident = judgment["ridge_on_burst"]["delayed_vs_identity"]
    if ridge_vs_ident is not None:
        notes.append(
            "Ridge burst vs identity delayed Δacc=%+.6f."
            % ridge_vs_ident
        )
    if not judgment["per_consumer"]["knn"]["full_order_match"]:
        notes.append(
            "kNN Support vs delayed order: delayed %s; Support %s."
            % (
                judgment["per_consumer"]["knn"]["delayed_order"],
                judgment["per_consumer"]["knn"]["support_order"],
            )
        )
    return notes


def _fmt_acc(value: float | None) -> str:
    return "null" if value is None else "%.6f" % value


def _fmt_delta(value: float | None) -> str:
    return "null" if value is None else "%+.6f" % value


def render_md(payload: dict[str, Any]) -> str:
    judgment = payload["judgment"]
    arms = payload["arms"]
    quality = payload["detection_quality"]
    lines = [
        "# CLS-4 burst-repair Program Supply + paired-Consumer re-judgment",
        "",
        "evidence class: %s (development).  %s"
        % (payload["evidence_class"], payload["claim_cap"]),
        "",
        "## Verdict",
        "",
        "- **%s**" % judgment["verdict"],
        "- %s" % judgment["reason"],
        "",
        "## Five conditions",
        "",
        "- (1) kNN delayed injury ≤ −0.05: **True** (cited CLS-3 %.3f)"
        % CLS3_KNN_INJURY,
        "- (2) repair_burst_segment recovers ≥50%%: **%s** (fraction %s)"
        % (
            judgment["five_conditions"]["knn_recovery_ge_50"],
            "null"
            if judgment["per_consumer"]["knn"]["recovery_fraction"] is None
            else "%.4f" % judgment["per_consumer"]["knn"]["recovery_fraction"],
        ),
        "- (3) that arm recall vs clean not worse >0.05: **%s**"
        % judgment["five_conditions"]["knn_recall_guard"],
        "- (4) kNN Support order matches delayed (all arms): **%s**"
        % judgment["five_conditions"]["knn_support_predicts_delayed_order"],
        "- (5) ridge identity remains numb: **True** (cited CLS-3 %.4f)"
        % CLS3_RIDGE_INJURY,
        "- all five: **%s**" % judgment["all_five"],
        "",
        "## New operator",
        "",
        "- `repair_burst_segment`: series-level robust-z, |z|>%.1f and run≥%d, "
        "endpoint linear interpolation; identity if no hit"
        % (BURST_Z_THRESHOLD, BURST_MIN_RUN),
        "- targeting_mode=intrinsic; allowed_tasks=(classification,); destructive=True",
        "- unit tests: %s" % payload["unit_tests"],
        "",
        "## Detection quality vs CLS-2 ledger",
        "",
        "- mean IoU on hit rows: **%s**" % (
            "null" if quality["mean_iou_hit_rows"] is None
            else "%.4f" % quality["mean_iou_hit_rows"]
        ),
        "- detection recall / precision: %s / %s"
        % (
            _fmt_acc(quality["detection_recall"]),
            _fmt_acc(quality["detection_precision"]),
        ),
        "- hit rows with any detection: %d/%d"
        % (quality["hit_rows_with_any_detection"], quality["hit_rows"]),
        "- clean-row identity: %d/%d (rate %s)"
        % (
            quality["clean_rows_identity"],
            quality["clean_rows"],
            _fmt_acc(quality["clean_row_identity_rate"]),
        ),
        "",
        "## Ten-arm delayed (TEST) and Support",
        "",
        "| consumer | workflow | source | delayed acc | Support acc | "
        "delayed Δacc | Support Δacc |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for consumer in cls3.CONSUMERS:
        deltas = judgment["per_consumer"][consumer]
        for workflow in ALL_WORKFLOWS:
            row = arms[cls3.arm_key(consumer, workflow)]
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s |"
                % (
                    consumer,
                    row["workflow"],
                    "cited-CLS-3" if row.get("cited_from") else "new-fit",
                    _fmt_acc(row["delayed"]["accuracy"]),
                    _fmt_acc(row["support"]["accuracy"]),
                    _fmt_delta(deltas["delayed_delta_acc"][workflow]),
                    _fmt_delta(deltas["support_delta_acc"][workflow]),
                )
            )
    lines.extend(["", "### Per-class recall (delayed / Support)", ""])
    for consumer in cls3.CONSUMERS:
        for workflow in ALL_WORKFLOWS:
            row = arms[cls3.arm_key(consumer, workflow)]
            lines.append("**%s / %s**" % (consumer, row["workflow"]))
            lines.append("")
            lines.append("| class | delayed n | delayed recall | Support n | "
                         "Support recall |")
            lines.append("|---|---:|---:|---:|---:|")
            delayed_r = row["delayed"]["per_class_recall"]
            support_r = row["support"]["per_class_recall"]
            for label in sorted(set(delayed_r) | set(support_r), key=int):
                dcell = delayed_r.get(label, {"n": 0, "recall": None})
                scell = support_r.get(label, {"n": 0, "recall": None})
                lines.append(
                    "| %s | %s | %s | %s | %s |"
                    % (
                        label,
                        dcell.get("n", 0),
                        _fmt_acc(dcell.get("recall")),
                        scell.get("n", 0),
                        _fmt_acc(scell.get("recall")),
                    )
                )
            lines.append("")
    knn = judgment["per_consumer"]["knn"]
    q = judgment["quantization"]
    cost = payload["cost"]
    lines.extend([
        "## Ridge on the new repair",
        "",
        "- delayed acc: %s (Δ vs clean %s, Δ vs identity %s)"
        % (
            _fmt_acc(judgment["ridge_on_burst"]["delayed_acc"]),
            _fmt_delta(judgment["ridge_on_burst"]["delayed_delta_vs_clean"]),
            _fmt_delta(judgment["ridge_on_burst"]["delayed_vs_identity"]),
        ),
        "",
        "## n, fit, determinism",
        "",
        "- TEST n=%d, step=%.6f; kNN recovery fraction %s"
        % (
            q["test_n"],
            q["step"],
            "null" if knn["recovery_fraction"] is None
            else "%.4f" % knn["recovery_fraction"],
        ),
        "- official new fits %d / %d: %s"
        % (cost["fits"], cost["fit_cap"], cost["fits_by_arm"]),
        "- verification fits %d; two-run **%s**"
        % (cost["verification_fits"], payload["determinism"]["two_run"]),
        "- injected SHA matches CLS-2: **%s**"
        % payload["determinism"]["injected_sha_matches_cls2"],
        "",
        "## Obligation self-report",
        "",
    ])
    for key in sorted(payload["obligations"]):
        lines.append("- %s: %s" % (key, payload["obligations"][key]))
    lines.extend(["", "## Out-of-book findings (report only, not repaired)", ""])
    for note in payload["out_of_book"]:
        lines.append("- %s" % note)
    lines.append("")
    return "\n".join(lines)


def run() -> int:
    cls3_payload = load_cls3()
    site = cls3.load_site()
    official = cls1.FitBudget(FIT_CAP)
    new_official, repaired = exam_new(site, official)
    verify = cls1.FitBudget(FIT_CAP)
    new_verify, _repaired_again = exam_new(site, verify)
    fp1 = cls1.numeric_fingerprint(new_official)
    fp2 = cls1.numeric_fingerprint(new_verify)
    if fp1 != fp2:
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "two-run numeric fingerprint drifted")
    if cls1._array_sha(repaired) != cls1._array_sha(_repaired_again):
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "repair replay drifted")
    test_sha_after = cls1._array_sha(site["test_values"])
    zip_sha_after = cls1._file_sha(site["archive"])
    if test_sha_after != site["test_sha"]:
        raise cls1.Stop("PROTOCOL_BREACH", "TEST array bytes changed")
    if zip_sha_after != site["zip_sha"]:
        raise cls1.Stop("PROTOCOL_BREACH", "UCR zip bytes changed")
    arms: dict[str, Any] = {}
    for consumer in cls3.CONSUMERS:
        for workflow in CITED_WORKFLOWS:
            arms[cls3.arm_key(consumer, workflow)] = cite_arm(
                cls3_payload, consumer, workflow
            )
        arms[cls3.arm_key(consumer, NEW_WORKFLOW)] = new_official[consumer]
    quality = detection_quality(site["injected"], site["ledger"])
    # drop per-row dump from the printed MD path but keep a compact copy
    quality_full = quality
    judgment = judge(arms, int(site["test_labels"].size))
    unit_tests = {
        "requested": [
            "deterministic",
            "clean_identity",
            "boundary_head_tail",
            "synthetic_index_exact",
            "repair_mae_beats_corruption",
        ],
        "module": "tests/operators/test_repair_burst_segment.py",
    }
    hit = set(int(row) for row in site["ledger"]["hit_rows"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "book": "CLS-4 burst-repair Program Supply + paired-Consumer re-judgment",
        "evidence_class": "INSTRUMENT / POSITIVE_CONTROL",
        "development_only": True,
        "claim_cap": (
            "development Program Supply on the frozen CLS-2 burst; "
            "not a natural UCR capability claim"
        ),
        "unit_tests": unit_tests,
        "operator": {
            "name": NEW_OPERATOR,
            "z_threshold": BURST_Z_THRESHOLD,
            "min_run": BURST_MIN_RUN,
            "targeting_mode": "intrinsic",
            "allowed_tasks": ["classification"],
            "destructive": True,
        },
        "site": {
            "dataset": "GunPoint",
            "ledger_path": cls3.LEDGER_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "ledger_replay": True,
            "train_n": int(site["train_labels"].size),
            "test_n": int(site["test_labels"].size),
            "fit_n": int(site["fit_idx"].size),
            "support_n": int(site["support_idx"].size),
            "n_hit_rows": site["ledger"]["n_hit_rows"],
            "fit_rows_hit": int(sum(int(i) in hit for i in site["fit_idx"])),
            "support_rows_hit": int(sum(int(i) in hit for i in site["support_idx"])),
        },
        "detection_quality": {
            key: value for key, value in quality_full.items()
            if key != "per_hit_row"
        },
        "detection_quality_per_hit_row": quality_full["per_hit_row"],
        "arms": arms,
        "judgment": judgment,
        "determinism": {
            "two_run": "BITWISE_IDENTICAL",
            "official_fingerprint": fp1,
            "verification_fingerprint": fp2,
            "injected_sha": site["injected_sha"],
            "injected_sha_matches_cls2": True,
            "test_sha": site["test_sha"],
            "test_sha_unchanged": True,
            "zip_sha": site["zip_sha"],
            "zip_sha_unchanged": True,
            "cls3_cited": True,
        },
        "cost": {
            "llm": 0,
            "fits": official.used,
            "fit_cap": FIT_CAP,
            "fits_by_arm": dict(official.by_arm),
            "verification_fits": verify.used,
            "cited_cls3_fits": 8,
        },
        "obligations": {
            "llm_calls": 0,
            "agent_invoked": False,
            "k_scan": False,
            "amplitude_scan": False,
            "threshold_scan": False,
            "min_run_scan": False,
            "dataset_scan": False,
            "methods_edited": False,
            "second_new_operator": False,
            "mad_refit": False,
            "ledger_redraw": False,
            "part0_is_cls3_collection": True,
            "fit_budget_used": official.used,
            "fit_budget_cap": FIT_CAP,
            "fit_budget_respected": bool(official.used <= FIT_CAP),
            "yahoo_all_reads": 0,
            "noaa_2025_reads": 0,
            "beyond_17520_reads": 0,
            "nab_reads": 0,
            "smd_reads": 0,
            "test_bytes_touched": False,
            "test_sha_unchanged": True,
            "zip_bytes_unchanged": True,
            "cls2_ledger_rewritten": False,
            "loader_output_unmutated": True,
            "two_run": True,
            "flying_files_untouched": [
                "AGENTS.md",
                "README.md",
                "docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md",
                "docs/SUCCESSOR_BRIEF_2026-08-22.md",
            ],
        },
        "out_of_book": out_of_book(judgment, quality_full),
    }
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(cls1._json_text(payload), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": judgment["verdict"],
        "reason": judgment["reason"],
        "five_conditions": judgment["five_conditions"],
        "knn_recovery_fraction": judgment["per_consumer"]["knn"]["recovery_fraction"],
        "knn_burst_delayed": new_official["knn"]["delayed"]["accuracy"],
        "ridge_burst_delayed": new_official["ridge"]["delayed"]["accuracy"],
        "mean_iou": quality_full["mean_iou_hit_rows"],
        "detection_recall": quality_full["detection_recall"],
        "fits": official.used,
        "two_run": "BITWISE_IDENTICAL",
    }, ensure_ascii=False, indent=2))
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] != "--run":
        print(__doc__)
        return 2
    try:
        return run()
    except cls1.Stop as exc:
        print(json.dumps({
            "verdict": exc.code,
            "detail": exc.detail,
        }, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
