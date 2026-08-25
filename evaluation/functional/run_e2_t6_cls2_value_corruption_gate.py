"""CLS-2 -- classification qualification gate, value-corruption family.

Train-only missingness is closed.  The new defect is one contiguous
noisy burst per hit row (values stay finite).  Identity therefore has
no drop-row escape: it fits the corrupted substrate as-is.

Menu, frozen after registry reconnaissance:
  identity + hampel_filter + outlier_mad
Hampel is mandated.  outlier_mad is the second classification-legal
repair whose mechanism (global robust clip) matches a 5-sigma burst
against an 80–85% clean majority, and is distinct from Hampel's local
window (which inflates inside a long burst).

One substrate (GunPoint).  No ladder.  0 LLM.

Usage:
  python evaluation/functional/run_e2_t6_cls2_value_corruption_gate.py --run
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

from SelfEvolvingHarnessTS.contracts.observables import (  # noqa: E402
    OUTLIER_Z_THRESHOLD,
)
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    classification_global_coarse_task_quality_contract_v1,
    classification_task_context_v1,
    classification_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    _window_summary,
    extract_public_features,
)
from SelfEvolvingHarnessTS.operators.registry import (  # noqa: E402
    OPERATOR_METADATA,
    OPERATOR_NAMES,
)

import run_e2_t6_cls1_qualification_gate as cls1  # noqa: E402
import run_e2_task_context_label_evidence_witness as witness  # noqa: E402

PROTOCOL_VERSION = "t6_cls2_value_corruption_gate_v1"
RUN_ID = "cls2_v1"
DATASET = "GunPoint"
ROW_FRACTION = 0.50
SEG_FRAC_MIN = 0.15
SEG_FRAC_MAX = 0.20
NOISE_SIGMA_MULT = 5.0
SEED_SUPPORT = cls1.SEED_SUPPORT
SEED_INJECT = 202608254
FIT_CAP = cls1.FIT_CAP
INJURY_BAR = cls1.INJURY_BAR
RECOVERY_FRACTION_BAR = cls1.RECOVERY_FRACTION_BAR
RECALL_HARM_BAR = cls1.RECALL_HARM_BAR

ARMS = (
    "clean_reference",
    "corrupted_identity",
    "corrupted_hampel",
    "corrupted_outlier_mad",
)
REPAIR_ARMS = ("corrupted_hampel", "corrupted_outlier_mad")
OPERATOR_FOR_ARM = {
    "corrupted_hampel": "hampel_filter",
    "corrupted_outlier_mad": "outlier_mad",
}
IDENTITY_ARM = "corrupted_identity"
PUBLIC_COMPARE_KEYS = (
    "missing_fraction",
    "longest_missing_run_fraction",
    "local_robust_z_peak",
    "estimated_region_start_fraction",
    "estimated_region_end_fraction",
    "outlier_region_end_fraction",
    "level_region_fraction",
    "level_excursion_score",
    "estimated_level_offset",
    "period_change_score",
    "period_reliability",
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_cls2_value_corruption_gate.json"
OUT_MD = E2 / "t6_cls2_value_corruption_gate.md"
SCRATCH = PROJECT_ROOT / "_scratch" / "cls2" / RUN_ID


def _is_classification_legal(name: str) -> bool:
    meta = OPERATOR_METADATA[name]
    if meta.get("is_alias"):
        return False
    return "classification" in tuple(meta.get("allowed_tasks") or ())


def menu_reconnaissance() -> dict[str, Any]:
    smoothing_repair: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for name in OPERATOR_NAMES:
        if not _is_classification_legal(name):
            continue
        meta = OPERATOR_METADATA[name]
        row = {
            "name": name,
            "category": meta["category"],
            "tags": list(meta.get("tags") or []),
            "targeting_mode": meta.get("targeting_mode"),
            "destructive": bool(meta.get("destructive")),
            "preserves_observed": bool(meta.get("preserves_observed")),
            "shape_changing": bool(meta.get("shape_changing")),
        }
        if meta.get("shape_changing") or name in {
            "znorm", "minmax_norm", "resample_uniform",
        }:
            row["why_not_menu"] = "scale/align/shape-change, not in-place repair"
            excluded.append(row)
            continue
        if meta["category"] == "impute":
            row["why_not_menu"] = (
                "preserves_observed imputer is a no-op on finite value corruption"
            )
            excluded.append(row)
            continue
        smoothing_repair.append(row)
    return {
        "classification_legal_smoothing_or_repair": smoothing_repair,
        "classification_legal_excluded_from_menu": excluded,
        "mandated": "hampel_filter",
        "selected_w2": "outlier_mad",
        "w2_reason": (
            "Contiguous 5×std burst on 15–20% of a z-normed row is a tail "
            "against the remaining 80–85% clean mass.  outlier_mad "
            "(intrinsic, global robust clip, k=3.5) is the matching "
            "family: it can see the burst without a window longer than "
            "the burst.  denoise_median default window=5 cannot eat a "
            "23–30 point GunPoint burst.  Hampel's window=7 local MAD "
            "inflates inside the burst, so the second slot must not "
            "duplicate that local-window family.  winsorize/outlier_iqr "
            "are the same global-clip family as outlier_mad (registry "
            "docstring); one representative is enough.  "
            "repair_level_shift targets a two-boundary level geometry, "
            "not additive burst noise."
        ),
    }


def select_hit_rows(
    labels: np.ndarray,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    rng = np.random.RandomState(seed)
    hit: list[int] = []
    by_class: dict[str, Any] = {}
    for label in sorted(int(value) for value in np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        n_hit = int(round(ROW_FRACTION * len(indices)))
        if n_hit < 1 or n_hit >= len(indices):
            raise cls1.Stop(
                "INSTRUMENT_UNREADABLE",
                "class %d n=%d cannot take stratified %.0f%% hits"
                % (label, int(len(indices)), 100 * ROW_FRACTION),
            )
        chosen = np.sort(rng.choice(indices, size=n_hit, replace=False))
        by_class[str(label)] = {
            "n": int(len(indices)),
            "n_hit": int(n_hit),
            "rows": [int(index) for index in chosen],
        }
        hit.extend(int(index) for index in chosen)
    return sorted(set(hit)), by_class


def inject_burst_noise(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.RandomState(seed)
    injected = np.asarray(values, dtype=np.float64).copy()
    n_rows, length = injected.shape
    length_lo = int(np.ceil(SEG_FRAC_MIN * length))
    length_hi = int(np.floor(SEG_FRAC_MAX * length))
    if length_lo < 1 or length_hi < length_lo or length_hi >= length:
        raise cls1.Stop(
            "INSTRUMENT_UNREADABLE",
            "L=%d cannot host a 15–20%% contiguous burst" % length,
        )
    hit_rows, by_class = select_hit_rows(labels, seed)
    row_records: list[dict[str, Any]] = []
    for row in hit_rows:
        seg_len = int(rng.randint(length_lo, length_hi + 1))
        start = int(rng.randint(0, length - seg_len + 1))
        row_std = float(np.std(values[row]))
        if not np.isfinite(row_std) or row_std <= 1e-12:
            raise cls1.Stop(
                "INSTRUMENT_UNREADABLE",
                "row %d has non-usable std" % row,
            )
        sigma = NOISE_SIGMA_MULT * row_std
        noise = rng.normal(0.0, sigma, size=seg_len).astype(np.float64)
        injected[row, start:start + seg_len] = (
            values[row, start:start + seg_len] + noise
        )
        row_records.append({
            "row": int(row),
            "label": int(labels[row]),
            "start": start,
            "length": seg_len,
            "fraction": float(seg_len / length),
            "row_std": row_std,
            "sigma": sigma,
            "noise": [float(value) for value in noise],
        })
    clean_rows = [int(row) for row in range(n_rows) if row not in set(hit_rows)]
    for row in clean_rows:
        if not np.array_equal(injected[row], values[row]):
            raise cls1.Stop("PROTOCOL_BREACH", "clean row %d was mutated" % row)
    if not np.isfinite(injected).all():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "injection produced non-finite values")
    ledger = {
        "form": "row_subset_one_contiguous_additive_burst",
        "row_fraction": ROW_FRACTION,
        "segment_fraction": [SEG_FRAC_MIN, SEG_FRAC_MAX],
        "noise_sigma_multiplier": NOISE_SIGMA_MULT,
        "seed": int(seed),
        "n_rows": int(n_rows),
        "series_length": int(length),
        "segment_length_bounds": [length_lo, length_hi],
        "n_hit_rows": int(len(hit_rows)),
        "n_clean_rows": int(len(clean_rows)),
        "hit_rows": hit_rows,
        "clean_rows": clean_rows,
        "by_class": by_class,
        "rows": row_records,
        "distinct_from_w43": (
            "W43 is isolated-point class-conditioned impulse; this is one "
            "contiguous segment of additive Gaussian burst noise"
        ),
    }
    return injected, ledger


def public_row(values: np.ndarray) -> dict[str, float]:
    features = dict(extract_public_features(values, task_kind="classification"))
    return {key: float(features[key]) for key in PUBLIC_COMPARE_KEYS}


def observation_visibility(
    clean: np.ndarray,
    corrupted: np.ndarray,
    hit_rows: list[int],
) -> dict[str, Any]:
    clean_cohort = _window_summary(
        [clean[row] for row in range(clean.shape[0])], calendar_period=4
    )
    dirty_cohort = _window_summary(
        [corrupted[row] for row in range(corrupted.shape[0])], calendar_period=4
    )
    clean_rows = [public_row(clean[row]) for row in range(clean.shape[0])]
    dirty_rows = [public_row(corrupted[row]) for row in range(corrupted.shape[0])]
    hit = set(hit_rows)

    def _mean(rows: list[dict[str, float]], key: str, only_hit: bool) -> float:
        values = [
            row[key] for index, row in enumerate(rows)
            if (index in hit) == only_hit or not only_hit
        ]
        picked = [
            row[key] for index, row in enumerate(rows)
            if (not only_hit) or (index in hit)
        ]
        del values
        return float(np.mean(picked)) if picked else float("nan")

    def _max(rows: list[dict[str, float]], key: str, only_hit: bool) -> float:
        picked = [
            row[key] for index, row in enumerate(rows)
            if (not only_hit) or (index in hit)
        ]
        return float(np.max(picked)) if picked else float("nan")

    z_clean_hit = [clean_rows[row]["local_robust_z_peak"] for row in hit_rows]
    z_dirty_hit = [dirty_rows[row]["local_robust_z_peak"] for row in hit_rows]
    z_rise = [
        dirty > clean_z + 1e-12
        for dirty, clean_z in zip(z_dirty_hit, z_clean_hit)
    ]
    visible = bool(
        np.mean(z_dirty_hit) > np.mean(z_clean_hit)
        and max(z_dirty_hit) >= OUTLIER_Z_THRESHOLD
    )
    feature_deltas = {}
    for key in PUBLIC_COMPARE_KEYS:
        feature_deltas[key] = {
            "clean_mean_all": _mean(clean_rows, key, False),
            "corrupted_mean_all": _mean(dirty_rows, key, False),
            "clean_mean_hit": _mean(clean_rows, key, True),
            "corrupted_mean_hit": _mean(dirty_rows, key, True),
            "clean_max_hit": _max(clean_rows, key, True),
            "corrupted_max_hit": _max(dirty_rows, key, True),
            "delta_mean_hit": (
                _mean(dirty_rows, key, True) - _mean(clean_rows, key, True)
            ),
        }
    return {
        "coverage_clean": float(clean_cohort["coverage"]),
        "coverage_corrupted": float(dirty_cohort["coverage"]),
        "max_missing_run_clean": int(clean_cohort["maximum_missing_run_length"]),
        "max_missing_run_corrupted": int(
            dirty_cohort["maximum_missing_run_length"]
        ),
        "missing_signal_present": bool(
            float(dirty_cohort["coverage"]) < 1.0
            or int(dirty_cohort["maximum_missing_run_length"]) > 0
        ),
        "outlier_z_threshold": OUTLIER_Z_THRESHOLD,
        "hit_rows_z_peak_rose": int(sum(z_rise)),
        "hit_rows": int(len(hit_rows)),
        "mean_z_peak_clean_hit": float(np.mean(z_clean_hit)),
        "mean_z_peak_corrupted_hit": float(np.mean(z_dirty_hit)),
        "max_z_peak_corrupted_hit": float(np.max(z_dirty_hit)),
        "value_corruption_visible_in_public_features": visible,
        "visibility_note": (
            "value corruption does not move coverage; visibility is "
            "carried by local_robust_z_peak / region features, which is "
            "what a future Fast Agent could read"
            if visible
            else "public features did not produce a readable outlier-scale "
            "shift on hit rows; a future Agent may not see this defect"
        ),
        "feature_deltas": feature_deltas,
    }


def run_arm(
    *,
    arm: str,
    site: dict[str, Any],
    budget: cls1.FitBudget,
) -> dict[str, Any]:
    fit_idx = site["fit_idx"]
    support_idx = site["support_idx"]
    labels = site["train_labels"]
    if arm == "clean_reference":
        fit_values = site["train_values"][fit_idx]
        support_values = site["train_values"][support_idx]
        workflow = "identity_on_clean"
    elif arm == IDENTITY_ARM:
        fit_values = site["injected"][fit_idx]
        support_values = site["injected"][support_idx]
        workflow = "identity_on_corrupted"
    else:
        operator = OPERATOR_FOR_ARM[arm]
        repaired = cls1.apply_operator(operator, site["injected"])
        fit_values = repaired[fit_idx]
        support_values = repaired[support_idx]
        workflow = operator
    drop_census = {
        "n_in": int(fit_values.shape[0]),
        "n_kept": int(fit_values.shape[0]),
        "n_dropped": 0,
        "classes_kept": [
            int(label) for label in sorted(np.unique(labels[fit_idx]))
        ],
        "note": "value corruption stays finite; identity does not drop rows",
    }
    model, fit_info = cls1.fit_ridge(budget, arm, fit_values, labels[fit_idx])
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
        "arm": arm,
        "workflow": workflow,
        "fit": fit_info,
        "drop_census_fit": drop_census,
        "drop_census_support": {
            "n_in": int(support_idx.size),
            "n_kept": int(support_idx.size),
            "n_dropped": 0,
        },
        "delayed": delayed,
        "support": support,
    }


def exam(site: dict[str, Any], budget: cls1.FitBudget) -> dict[str, Any]:
    return {arm: run_arm(arm=arm, site=site, budget=budget) for arm in ARMS}


def judge(arms: dict[str, Any], test_n: int) -> dict[str, Any]:
    step = 1.0 / float(test_n)
    clean_d = cls1._acc(arms["clean_reference"], "delayed")
    ident_d = cls1._acc(arms[IDENTITY_ARM], "delayed")
    clean_s = cls1._acc(arms["clean_reference"], "support")
    delayed_delta = {
        arm: cls1._delta(cls1._acc(arms[arm], "delayed"), clean_d) for arm in ARMS
    }
    support_delta = {
        arm: cls1._delta(cls1._acc(arms[arm], "support"), clean_s) for arm in ARMS
    }
    injury = delayed_delta[IDENTITY_ARM]
    injury_readable = injury is not None and injury <= INJURY_BAR
    recoveries: dict[str, Any] = {}
    legal_headroom = False
    best_repair: str | None = None
    best_recovery = None
    if injury is not None and injury < 0.0 and clean_d is not None and ident_d is not None:
        injury_amount = clean_d - ident_d
        for arm in REPAIR_ARMS:
            repair_d = cls1._acc(arms[arm], "delayed")
            if repair_d is None:
                recoveries[arm] = {"status": "UNSCORED"}
                continue
            recovered = repair_d - ident_d
            fraction = recovered / injury_amount if injury_amount > 0 else None
            clean_recalls = arms["clean_reference"]["delayed"]["per_class_recall"]
            repair_recalls = arms[arm]["delayed"]["per_class_recall"]
            recall_drops: dict[str, float | None] = {}
            recall_ok = True
            for label in clean_recalls:
                clean_r = clean_recalls[label]["recall"]
                repair_r = repair_recalls.get(label, {}).get("recall")
                drop = (
                    None if clean_r is None or repair_r is None
                    else float(clean_r - repair_r)
                )
                recall_drops[label] = drop
                if drop is not None and drop > RECALL_HARM_BAR:
                    recall_ok = False
            qualifies = (
                fraction is not None
                and fraction >= RECOVERY_FRACTION_BAR
                and recall_ok
            )
            recoveries[arm] = {
                "delayed_accuracy": repair_d,
                "recovered_acc": recovered,
                "recovery_fraction": fraction,
                "recall_drop_vs_clean": recall_drops,
                "recall_guard_ok": recall_ok,
                "qualifies": qualifies,
            }
            if qualifies:
                legal_headroom = True
            if fraction is not None and (
                best_recovery is None or fraction > best_recovery
            ):
                best_recovery = fraction
                best_repair = arm
    else:
        for arm in REPAIR_ARMS:
            recoveries[arm] = {"status": "INJURY_UNDEFINED"}
    delayed_order = cls1.rank_key(delayed_delta)
    support_order = cls1.rank_key(support_delta)
    full_order_match = (
        delayed_order is not None
        and support_order is not None
        and delayed_order == support_order
    )
    pair_direction = None
    if best_repair is not None:
        d_ident = delayed_delta[IDENTITY_ARM]
        d_best = delayed_delta[best_repair]
        s_ident = support_delta[IDENTITY_ARM]
        s_best = support_delta[best_repair]
        if None not in (d_ident, d_best, s_ident, s_best):
            pair_direction = (d_best > d_ident) == (s_best > s_ident)
    b2 = bool(full_order_match or pair_direction is True)
    if any(cls1._acc(arms[arm], "delayed") is None for arm in ARMS):
        verdict = "INSTRUMENT_UNREADABLE"
        reason = "at least one arm could not produce a delayed accuracy"
    elif not injury_readable:
        verdict = "INJURY_NOT_READABLE"
        reason = (
            "clean vs corrupted+identity delayed Δacc is %s; "
            "pre-registered bar is <= %.3f"
            % (injury, INJURY_BAR)
        )
    elif not legal_headroom:
        verdict = "NO_LEGAL_HEADROOM"
        reason = (
            "injury is readable but no repair arm recovered >=50% "
            "without a class-recall drop > 0.05"
        )
    elif not b2:
        verdict = "SUPPORT_NOT_PREDICTIVE"
        reason = (
            "B1 passed but Support Δacc order does not match delayed, "
            "including the identity / best-repair pair"
        )
    else:
        verdict = "CLS_GATE_PASSED"
        reason = "B1 injury + legal headroom and B2 Support direction passed"
    return {
        "verdict": verdict,
        "reason": reason,
        "quantization": {
            "test_n": int(test_n),
            "step": step,
            "injury_bar": INJURY_BAR,
            "injury_bar_abs_steps": abs(INJURY_BAR) / step,
            "injury_bar_above_floor": abs(INJURY_BAR) >= step,
            "recall_harm_bar_above_floor": RECALL_HARM_BAR >= step,
            "recovery_fraction_bar": RECOVERY_FRACTION_BAR,
        },
        "b1": {
            "injury_delta_acc": injury,
            "injury_readable": injury_readable,
            "injury_bar": INJURY_BAR,
            "legal_headroom": legal_headroom,
            "recoveries": recoveries,
            "best_repair_arm": best_repair,
            "best_recovery_fraction": best_recovery,
        },
        "b2": {
            "delayed_delta_acc": delayed_delta,
            "support_delta_acc": support_delta,
            "delayed_order": delayed_order,
            "support_order": support_order,
            "full_order_match": full_order_match,
            "identity_best_repair_direction_match": pair_direction,
            "passed": b2,
        },
    }


def load_site() -> dict[str, Any]:
    archive = PROJECT_ROOT / witness.DATA_DIR / ("%s.zip" % DATASET)
    if not archive.is_file():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "missing archive %s" % archive)
    zip_sha_before = cls1._file_sha(archive)
    train_values, train_labels = witness._load_split(np, archive, DATASET, "TRAIN")
    test_values, test_labels = witness._load_split(np, archive, DATASET, "TEST")
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "loader emitted non-finite values")
    fit_idx, support_idx = cls1.split_fit_support(train_labels, SEED_SUPPORT)
    injected, ledger = inject_burst_noise(
        train_values, train_labels, seed=SEED_INJECT
    )
    again, again_ledger = inject_burst_noise(
        train_values, train_labels, seed=SEED_INJECT
    )
    if cls1._json_text(ledger) != cls1._json_text(again_ledger):
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "injection ledger drifted")
    if cls1._array_sha(injected) != cls1._array_sha(again):
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "injected array drifted")
    reload_train, _labels = witness._load_split(np, archive, DATASET, "TRAIN")
    if not np.array_equal(train_values, reload_train):
        raise cls1.Stop("PROTOCOL_BREACH", "TRAIN memory copy drifted after inject")
    if np.array_equal(injected, train_values):
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "injection left TRAIN unchanged")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "injection_ledger.json").write_text(
        cls1._json_text(ledger), encoding="utf-8"
    )
    np.save(SCRATCH / "corrupted_held_in.npy", injected)
    np.save(SCRATCH / "clean_train.npy", train_values)
    if cls1._file_sha(archive) != zip_sha_before:
        raise cls1.Stop("PROTOCOL_BREACH", "UCR zip bytes changed during site build")
    return {
        "archive": archive,
        "zip_sha": zip_sha_before,
        "train_values": train_values,
        "train_labels": train_labels,
        "test_values": test_values,
        "test_labels": test_labels,
        "fit_idx": fit_idx,
        "support_idx": support_idx,
        "injected": injected,
        "ledger": ledger,
        "test_sha": cls1._array_sha(test_values),
        "train_sha": cls1._array_sha(train_values),
        "injected_sha": cls1._array_sha(injected),
    }


def dataset_choice() -> dict[str, Any]:
    return {
        "selected": DATASET,
        "rejected": "ECG200",
        "reason": (
            "GunPoint clean delayed acc was 0.82 in CLS-1-r2 versus chance "
            "~0.50 (0.32 drop room).  ECG200 clean was 0.80 versus a 0.64 "
            "majority floor (only 0.16 room), and r2 already showed that "
            "imbalance can manufacture a +0.02 identity gain.  The Consumer "
            "is raw || first difference: a 15–20% burst on L=150 corrupts "
            "23–30 raw coordinates plus two ~5σ difference spikes at the "
            "boundaries, more feature mass than ECG200 L=96 (14–19 points).  "
            "Injury bar 0.05 is 7.5 TEST steps here versus 5 on ECG200.  "
            "No substrate ladder."
        ),
        "no_ladder": True,
    }


def out_of_book(site: dict[str, Any], arms: dict[str, Any], vis: dict[str, Any]) -> list[str]:
    notes = [
        "Loader z-norms each TRAIN row, so 5×row std is 5.0 on the "
        "unit-variance series.  Amplitude was not rescaled.",
        "Hampel default window=7 may treat the burst interior as a new "
        "local regime; that is a mechanism risk, not a parameter scan.",
        "No missingness family operators were added.  Impute ops remain "
        "no-ops on this finite substrate.",
        "Part 0 collected CLS-replay only; this CLS-2 artifact stays "
        "uncommitted.",
    ]
    if not vis["missing_signal_present"]:
        notes.append(
            "As required: coverage stayed 1.0 and max_missing_run stayed 0 "
            "after value corruption."
        )
    clean_d = cls1._acc(arms["clean_reference"], "delayed")
    ident_d = cls1._acc(arms[IDENTITY_ARM], "delayed")
    if clean_d is not None and ident_d is not None:
        notes.append(
            "clean vs identity delayed Δacc=%+.6f (bar=−0.05, step=%.6f)."
            % (ident_d - clean_d, 1.0 / float(site["test_labels"].size))
        )
    return notes


def render_md(payload: dict[str, Any]) -> str:
    judgment = payload["judgment"]
    vis = payload["observation_visibility"]
    recon = payload["menu_reconnaissance"]
    arms = payload["arms"]
    deltas = vis["feature_deltas"]
    lines = [
        "# CLS-2 classification value-corruption qualification gate",
        "",
        "evidence class: %s (development).  %s"
        % (payload["evidence_class"], payload["claim_cap"]),
        "",
        "## Verdict",
        "",
        "- **%s**" % judgment["verdict"],
        "- %s" % judgment["reason"],
        "",
        "## Dataset choice",
        "",
        "- selected: **%s** (rejected %s; no ladder)"
        % (payload["dataset_choice"]["selected"], payload["dataset_choice"]["rejected"]),
        "- %s" % payload["dataset_choice"]["reason"],
        "",
        "## Menu reconnaissance",
        "",
        "- mandated: `%s`" % recon["mandated"],
        "- W2: **`%s`**" % recon["selected_w2"],
        "- %s" % recon["w2_reason"],
        "",
        "| name | category | tags | targeting_mode | destructive |",
        "|---|---|---|---|---|",
    ]
    for row in recon["classification_legal_smoothing_or_repair"]:
        lines.append(
            "| `%s` | %s | %s | %s | %s |"
            % (
                row["name"],
                row["category"],
                ",".join(row["tags"]) or "—",
                row["targeting_mode"],
                row["destructive"],
            )
        )
    lines.extend([
        "",
        "Impute / shape / scale operators were listed as classification-legal "
        "but excluded from the repair menu (see JSON).",
        "",
        "## Site",
        "",
        "- Consumer: ridge-raw-plus-difference-v1 (reused)",
        "- quality contract: classification-global-coarse-quality-v1",
        "- held-in = official TRAIN; TEST = delayed, byte-zero-touch",
        "- 50%% class-stratified hit rows; one contiguous burst, "
        "15–20%% of L; additive N(0, (5×row std)²); seed %d"
        % payload["site"]["inject_seed"],
        "- identity = fit the corrupted substrate (no row drop)",
        "- ledger: `%s`" % payload["site"]["ledger_path"],
        "- hit %d/%d held-in (fit %d/%d, Support %d/%d)"
        % (
            payload["site"]["n_hit_rows"],
            payload["site"]["train_n"],
            payload["site"]["fit_rows_hit"],
            payload["site"]["fit_n"],
            payload["site"]["support_rows_hit"],
            payload["site"]["support_n"],
        ),
        "",
        "## Observation visibility (value-corruption variant)",
        "",
        "- coverage clean/corrupted: %.6f / %.6f"
        % (vis["coverage_clean"], vis["coverage_corrupted"]),
        "- max_missing_run clean/corrupted: %d / %d"
        % (vis["max_missing_run_clean"], vis["max_missing_run_corrupted"]),
        "- missing signal present: **%s** (expected False)"
        % vis["missing_signal_present"],
        "- hit-row local_robust_z_peak mean: %.4f → %.4f (max corrupted %.4f)"
        % (
            vis["mean_z_peak_clean_hit"],
            vis["mean_z_peak_corrupted_hit"],
            vis["max_z_peak_corrupted_hit"],
        ),
        "- hit rows whose z-peak rose: %d/%d"
        % (vis["hit_rows_z_peak_rose"], vis["hit_rows"]),
        "- value corruption visible in public features: **%s**"
        % vis["value_corruption_visible_in_public_features"],
        "- %s" % vis["visibility_note"],
        "",
        "| feature | clean mean (hit) | corrupted mean (hit) | Δ |",
        "|---|---:|---:|---:|",
    ])
    for key in PUBLIC_COMPARE_KEYS:
        item = deltas[key]
        lines.append(
            "| `%s` | %.6f | %.6f | %+.6f |"
            % (
                key,
                item["clean_mean_hit"],
                item["corrupted_mean_hit"],
                item["delta_mean_hit"],
            )
        )
    lines.extend([
        "",
        "## Four-arm delayed (TEST) and Support",
        "",
        "| arm | workflow | n_fit (dropped) | delayed acc | Support acc | "
        "delayed Δacc | Support Δacc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        row = arms[arm]
        d = row["delayed"]["accuracy"]
        s = row["support"]["accuracy"]
        dd = judgment["b2"]["delayed_delta_acc"][arm]
        sd = judgment["b2"]["support_delta_acc"][arm]
        lines.append(
            "| %s | %s | %s (%s) | %s | %s | %s | %s |"
            % (
                arm,
                row["workflow"],
                row["fit"]["n_train"],
                row["drop_census_fit"]["n_dropped"],
                "null" if d is None else "%.6f" % d,
                "null" if s is None else "%.6f" % s,
                "null" if dd is None else "%+.6f" % dd,
                "null" if sd is None else "%+.6f" % sd,
            )
        )
    lines.extend(["", "### Per-class recall (delayed / Support)", ""])
    for arm in ARMS:
        row = arms[arm]
        lines.append("**%s**" % arm)
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
                    "null" if dcell.get("recall") is None
                    else "%.6f" % dcell["recall"],
                    scell.get("n", 0),
                    "null" if scell.get("recall") is None
                    else "%.6f" % scell["recall"],
                )
            )
        lines.append("")
    rec = judgment["b1"]["recoveries"]
    lines.extend([
        "## B1 / B2",
        "",
        "- injury Δacc: **%s** (readable=%s, bar=−0.05, %.2f steps)"
        % (
            "null" if judgment["b1"]["injury_delta_acc"] is None
            else "%+.6f" % judgment["b1"]["injury_delta_acc"],
            judgment["b1"]["injury_readable"],
            judgment["quantization"]["injury_bar_abs_steps"],
        ),
        "- legal headroom: **%s**; best repair %s (recovery %s)"
        % (
            judgment["b1"]["legal_headroom"],
            judgment["b1"]["best_repair_arm"],
            "null" if judgment["b1"]["best_recovery_fraction"] is None
            else "%.4f" % judgment["b1"]["best_recovery_fraction"],
        ),
        "- Support vs delayed full order: %s; identity/best-repair "
        "direction: %s; B2: **%s**"
        % (
            judgment["b2"]["full_order_match"],
            judgment["b2"]["identity_best_repair_direction_match"],
            judgment["b2"]["passed"],
        ),
        "",
        "| arm | recovery fraction | recall guard | qualifies |",
        "|---|---:|---|---|",
    ])
    for arm in REPAIR_ARMS:
        item = rec.get(arm, {})
        if "qualifies" not in item:
            lines.append("| %s | — | — | %s |" % (arm, item.get("status")))
            continue
        frac = item["recovery_fraction"]
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                arm,
                "null" if frac is None else "%.4f" % frac,
                item["recall_guard_ok"],
                item["qualifies"],
            )
        )
    q = judgment["quantization"]
    cost = payload["cost"]
    lines.extend([
        "",
        "## n, fit, determinism",
        "",
        "- TEST n=%d, step=%.6f; injury bar %.2f steps (above floor=%s)"
        % (
            q["test_n"],
            q["step"],
            q["injury_bar_abs_steps"],
            q["injury_bar_above_floor"],
        ),
        "- official fits %d / %d: %s"
        % (cost["fits"], cost["fit_cap"], cost["fits_by_arm"]),
        "- verification fits %d; two-run **%s**"
        % (cost["verification_fits"], payload["determinism"]["two_run"]),
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
    recon = menu_reconnaissance()
    task_context = classification_task_context_v1(
        task_spec=classification_task_spec_v1(
            downstream_model_class="ridge-raw-plus-difference-v1"
        ),
        quality_contract=classification_global_coarse_task_quality_contract_v1(),
    )
    site = load_site()
    vis = observation_visibility(
        site["train_values"], site["injected"], site["ledger"]["hit_rows"]
    )
    official = cls1.FitBudget(FIT_CAP)
    arms = exam(site, official)
    verify = cls1.FitBudget(FIT_CAP)
    again = exam(site, verify)
    fp1 = cls1.numeric_fingerprint(arms)
    fp2 = cls1.numeric_fingerprint(again)
    if fp1 != fp2:
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "two-run numeric fingerprint drifted")
    test_sha_after = cls1._array_sha(site["test_values"])
    zip_sha_after = cls1._file_sha(site["archive"])
    if test_sha_after != site["test_sha"]:
        raise cls1.Stop("PROTOCOL_BREACH", "TEST array bytes changed")
    if zip_sha_after != site["zip_sha"]:
        raise cls1.Stop("PROTOCOL_BREACH", "UCR zip bytes changed")
    judgment = judge(arms, int(site["test_labels"].size))
    hit = set(site["ledger"]["hit_rows"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "book": "CLS-2 classification value-corruption qualification gate",
        "evidence_class": "INSTRUMENT / POSITIVE_CONTROL",
        "development_only": True,
        "claim_cap": (
            "development positive control on an injected contiguous burst; "
            "not a natural UCR capability claim"
        ),
        "task_context": task_context.to_dict(),
        "dataset_choice": dataset_choice(),
        "menu_reconnaissance": recon,
        "site": {
            "dataset": DATASET,
            "archive": "%s/%s.zip" % (witness.DATA_DIR, DATASET),
            "inject_seed": SEED_INJECT,
            "support_seed": SEED_SUPPORT,
            "train_n": int(site["train_labels"].size),
            "test_n": int(site["test_labels"].size),
            "series_length": int(site["train_values"].shape[1]),
            "fit_n": int(site["fit_idx"].size),
            "support_n": int(site["support_idx"].size),
            "n_hit_rows": site["ledger"]["n_hit_rows"],
            "fit_rows_hit": int(sum(int(i) in hit for i in site["fit_idx"])),
            "support_rows_hit": int(sum(int(i) in hit for i in site["support_idx"])),
            "ledger_path": (
                (SCRATCH / "injection_ledger.json")
                .relative_to(PROJECT_ROOT)
                .as_posix()
            ),
            "identity_policy": "fit corrupted finite rows; no drop-row escape",
        },
        "observation_visibility": vis,
        "arms": arms,
        "judgment": judgment,
        "determinism": {
            "two_run": "BITWISE_IDENTICAL",
            "injection_replay_identical": True,
            "official_fingerprint": fp1,
            "verification_fingerprint": fp2,
            "test_sha": site["test_sha"],
            "test_sha_unchanged": True,
            "zip_sha": site["zip_sha"],
            "zip_sha_unchanged": True,
            "injected_sha": site["injected_sha"],
        },
        "cost": {
            "llm": 0,
            "fits": official.used,
            "fit_cap": FIT_CAP,
            "fits_by_arm": dict(official.by_arm),
            "verification_fits": verify.used,
        },
        "obligations": {
            "llm_calls": 0,
            "agent_invoked": False,
            "rate_scan": False,
            "amplitude_scan": False,
            "third_repair": False,
            "preregistered_gates_rewritten": False,
            "substrate_ladder": False,
            "part0_sha": "f1fe3a004c959934e8f4cea72df8107d45616e2e",
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
            "loader_output_unmutated": True,
            "injection_after_load": True,
            "clean_rows_untouched": True,
            "two_run": True,
            "flying_files_untouched": [
                "AGENTS.md",
                "README.md",
                "docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md",
                "docs/SUCCESSOR_BRIEF_2026-08-22.md",
            ],
        },
        "out_of_book": out_of_book(site, arms, vis),
    }
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(cls1._json_text(payload), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": judgment["verdict"],
        "reason": judgment["reason"],
        "dataset": DATASET,
        "w2": recon["selected_w2"],
        "delayed": {arm: arms[arm]["delayed"]["accuracy"] for arm in ARMS},
        "injury_delta_acc": judgment["b1"]["injury_delta_acc"],
        "legal_headroom": judgment["b1"]["legal_headroom"],
        "visibility": vis["value_corruption_visible_in_public_features"],
        "z_peak_hit": {
            "clean": vis["mean_z_peak_clean_hit"],
            "corrupted": vis["mean_z_peak_corrupted_hit"],
        },
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
