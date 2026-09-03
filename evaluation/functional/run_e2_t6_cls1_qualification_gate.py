"""CLS-1 -- classification-slice qualification gate.  0 LLM, deterministic.

One Consumer (in-service ridge-raw-plus-difference-v1), one held-in MCAR
point-missing defect (15%, single rate), identity + two registered impute
Workflows.  No Agent.  Official TEST / Query is a delayed scoring surface
only: zero injection, zero processing.

This is a development positive-control instrument, not a natural-UCR
capability reading.

Usage:
  python evaluation/functional/run_e2_t6_cls1_qualification_gate.py --run
"""
from __future__ import annotations

import hashlib
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

from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    classification_global_coarse_task_quality_contract_v1,
    classification_task_context_v1,
    classification_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import _MISSING_ONLY_OPS  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    _window_summary,
    extract_public_features,
)
from SelfEvolvingHarnessTS.operators.registry import get_operator  # noqa: E402
from sklearn.linear_model import RidgeClassifier  # noqa: E402

import run_e2_task_context_label_evidence_witness as witness  # noqa: E402

# =========================================================================== #
# frozen by the book
# =========================================================================== #
PROTOCOL_VERSION = "t6_cls1_qualification_gate_v1"
RUN_ID = "cls1_v1"

DATA_DIR = witness.DATA_DIR
RIDGE_ALPHA = witness.RIDGE_ALPHA
SUPPORT_FRACTION = witness.SUPPORT_FRACTION
MIN_SUPPORT_PER_CLASS = witness.MIN_SUPPORT_PER_CLASS

# GunPoint: TRAIN 24/26 vs ECG200 31/69; TEST n=150 vs 100.
DATASET = "GunPoint"
MISSING_RATE = 0.15
MISSING_FORM = "point_mcar"
SEED_INJECT = 20260825
SEED_SUPPORT = 2026082501
FIT_CAP = 50
INJURY_BAR = -0.05
RECOVERY_FRACTION_BAR = 0.50
RECALL_HARM_BAR = 0.05

ARMS = (
    "clean_reference",
    "injected_identity",
    "injected_impute_linear",
    "injected_impute_ema",
)
IMPUTE_ARMS = ("injected_impute_linear", "injected_impute_ema")
OPERATOR_FOR_ARM = {
    "injected_impute_linear": "impute_linear",
    "injected_impute_ema": "impute_ema",
}

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_cls1_qualification_gate.json"
OUT_MD = E2 / "t6_cls1_qualification_gate.md"
SCRATCH = PROJECT_ROOT / "_scratch" / "cls1" / RUN_ID


class Stop(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__("%s: %s" % (code, detail))
        self.code = code
        self.detail = detail


class FitBudget:
    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.used = 0
        self.by_arm: dict[str, int] = {}

    def spend(self, arm: str, n: int = 1) -> None:
        if self.used + n > self.cap:
            raise Stop("CONSUMER_FIT_BUDGET_EXCEEDED",
                       "classification fit budget exhausted at %d" % self.cap)
        self.used += n
        self.by_arm[arm] = self.by_arm.get(arm, 0) + n


def _json_text(doc: Any) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").copy()
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _features(values: np.ndarray) -> np.ndarray:
    return witness._features(np, values)


def split_fit_support(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    fit: list[int] = []
    support: list[int] = []
    for label in sorted(int(value) for value in np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        count = max(MIN_SUPPORT_PER_CLASS, int(round(SUPPORT_FRACTION * len(indices))))
        if count >= len(indices):
            raise Stop(
                "INSTRUMENT_UNREADABLE",
                "class %d has %d rows; cannot cut support=%d"
                % (label, int(len(indices)), count),
            )
        chosen = np.sort(rng.choice(indices, size=count, replace=False))
        selected = {int(index) for index in chosen}
        support.extend(sorted(selected))
        fit.extend(int(index) for index in indices if int(index) not in selected)
    return (
        np.asarray(sorted(fit), dtype=np.int64),
        np.asarray(sorted(support), dtype=np.int64),
    )


def inject_point_mcar(
    values: np.ndarray,
    *,
    rate: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.RandomState(seed)
    injected = np.asarray(values, dtype=np.float64).copy()
    n_rows, length = injected.shape
    n_miss = int(round(rate * length))
    if n_miss < 1:
        raise Stop("INSTRUMENT_UNREADABLE", "MCAR rate produced zero missing points")
    rows: list[dict[str, Any]] = []
    for row in range(n_rows):
        positions = np.sort(rng.choice(length, size=n_miss, replace=False))
        injected[row, positions] = np.nan
        rows.append({
            "row": int(row),
            "n_missing": int(positions.size),
            "indices": [int(index) for index in positions],
        })
    ledger = {
        "form": MISSING_FORM,
        "rate": float(rate),
        "seed": int(seed),
        "n_rows": int(n_rows),
        "series_length": int(length),
        "n_missing_per_row": int(n_miss),
        "total_missing": int(n_rows * n_miss),
        "rows": rows,
    }
    return injected, ledger


def missing_observation(values: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float64)
    series = [matrix[row] for row in range(matrix.shape[0])]
    cohort = _window_summary(series, calendar_period=4)
    coverage = float(cohort["coverage"])
    max_run = int(cohort["maximum_missing_run_length"])
    per_series: list[dict[str, Any]] = []
    for row, series_values in enumerate(series):
        features = dict(extract_public_features(series_values, task_kind="classification"))
        per_series.append({
            "row": int(row),
            "missing_fraction": float(features["missing_fraction"]),
            "longest_missing_run_fraction": float(
                features["longest_missing_run_fraction"]),
        })
    missing_signal = coverage < 1.0 or max_run > 0
    return {
        "recent.coverage": coverage,
        "recent.maximum_missing_run_length": max_run,
        "missing_run_count": int(cohort["missing_run_count"]),
        "missing_signal_present": bool(missing_signal),
        "fast_agent_missing_only_ops": sorted(_MISSING_ONLY_OPS),
        "impute_ops_would_be_skipped": bool(not missing_signal),
        "per_series_public_features": per_series,
        "mean_missing_fraction": float(np.mean([
            row["missing_fraction"] for row in per_series
        ])),
    }


def per_class_recall(pred: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in sorted(int(value) for value in np.unique(labels)):
        mask = labels == label
        n_label = int(mask.sum())
        out[str(label)] = {
            "n": n_label,
            "recall": (
                float(np.mean(pred[mask] == labels[mask])) if n_label else None
            ),
        }
    return out


def score_model(
    model: RidgeClassifier,
    values: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    pred = model.predict(_features(values))
    return {
        "accuracy": float(np.mean(pred == labels)),
        "n": int(labels.size),
        "per_class_recall": per_class_recall(pred, labels),
        "scored": True,
        "reason": None,
    }


def drop_nan_rows(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    finite = np.isfinite(values).all(axis=1)
    kept = int(finite.sum())
    census = {
        "n_in": int(values.shape[0]),
        "n_kept": kept,
        "n_dropped": int(values.shape[0]) - kept,
        "classes_kept": (
            [int(label) for label in sorted(np.unique(labels[finite]))]
            if kept else []
        ),
    }
    return values[finite], labels[finite], census


def apply_operator(name: str, values: np.ndarray) -> np.ndarray:
    operator = get_operator(name)
    repaired = np.asarray(values, dtype=np.float64).copy()
    for row in range(repaired.shape[0]):
        repaired[row] = np.asarray(operator(repaired[row]), dtype=np.float64)
    if not np.isfinite(repaired).all():
        raise Stop("INSTRUMENT_UNREADABLE",
                   "%s left non-finite values" % name)
    return repaired


def fit_ridge(
    budget: FitBudget,
    arm: str,
    train_values: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[RidgeClassifier | None, dict[str, Any]]:
    n_rows = int(train_values.shape[0])
    classes = sorted(int(label) for label in np.unique(train_labels)) if n_rows else []
    if n_rows < 2 or len(classes) < 2:
        return None, {
            "fit": False,
            "reason": "NO_USABLE_TRAINING_ROWS",
            "n_train": n_rows,
            "classes": classes,
        }
    if not np.isfinite(train_values).all():
        return None, {
            "fit": False,
            "reason": "NON_FINITE_TRAINING_ROWS",
            "n_train": n_rows,
            "classes": classes,
        }
    budget.spend(arm)
    model = RidgeClassifier(alpha=RIDGE_ALPHA)
    model.fit(_features(train_values), train_labels)
    return model, {
        "fit": True,
        "reason": None,
        "n_train": n_rows,
        "classes": classes,
    }


def empty_score(reason: str, n: int = 0) -> dict[str, Any]:
    return {
        "accuracy": None,
        "n": int(n),
        "per_class_recall": {},
        "scored": False,
        "reason": reason,
    }


def load_site() -> dict[str, Any]:
    archive = PROJECT_ROOT / DATA_DIR / ("%s.zip" % DATASET)
    if not archive.is_file():
        raise Stop("INSTRUMENT_UNREADABLE", "missing archive %s" % archive)
    zip_sha_before = _file_sha(archive)
    train_values, train_labels = witness._load_split(np, archive, DATASET, "TRAIN")
    test_values, test_labels = witness._load_split(np, archive, DATASET, "TEST")
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise Stop("INSTRUMENT_UNREADABLE", "loader emitted non-finite values")
    fit_idx, support_idx = split_fit_support(train_labels, SEED_SUPPORT)
    injected, ledger = inject_point_mcar(
        train_values, rate=MISSING_RATE, seed=SEED_INJECT
    )
    if np.isfinite(injected).all():
        raise Stop("INSTRUMENT_UNREADABLE", "injection produced no missing values")
    if not np.array_equal(train_values, witness._load_split(
            np, archive, DATASET, "TRAIN")[0]):
        raise Stop("PROTOCOL_BREACH", "TRAIN memory copy drifted after inject")
    test_sha = _array_sha(test_values)
    train_sha = _array_sha(train_values)
    injected_sha = _array_sha(injected)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "injection_ledger.json").write_text(
        _json_text(ledger), encoding="utf-8"
    )
    np.save(SCRATCH / "injected_held_in.npy", injected)
    np.save(SCRATCH / "clean_train.npy", train_values)
    if _file_sha(archive) != zip_sha_before:
        raise Stop("PROTOCOL_BREACH", "UCR zip bytes changed during site build")
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
        "test_sha": test_sha,
        "train_sha": train_sha,
        "injected_sha": injected_sha,
    }


def dataset_choice_reason(site: dict[str, Any]) -> dict[str, Any]:
    train_labels = site["train_labels"]
    test_labels = site["test_labels"]
    train_counts = {
        str(label): int(np.count_nonzero(train_labels == label))
        for label in sorted(int(value) for value in np.unique(train_labels))
    }
    test_counts = {
        str(label): int(np.count_nonzero(test_labels == label))
        for label in sorted(int(value) for value in np.unique(test_labels))
    }
    return {
        "selected": DATASET,
        "rejected": "ECG200",
        "reason": (
            "GunPoint TRAIN is class-balanced (24/26 after official labels "
            "mapped 1,2 -> 0,1) against ECG200 TRAIN 31/69; TEST n=150 "
            "gives quantization floor 1/150≈0.00667 so the 0.05 injury bar "
            "is 7.5 steps, versus ECG200 TEST n=100 (floor 0.01, 5 steps). "
            "Both zips are already on the in-service loader; Support 30% "
            "still leaves >=7 series per class."
        ),
        "train_n": int(train_labels.size),
        "test_n": int(test_labels.size),
        "series_length": int(site["train_values"].shape[1]),
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "ecg200_probe": {
            "train_n": 100,
            "test_n": 100,
            "series_length": 96,
            "train_class_counts_official": {"-1": 31, "1": 69},
            "test_class_counts_official": {"-1": 36, "1": 64},
        },
    }


def run_arm(
    *,
    arm: str,
    site: dict[str, Any],
    budget: FitBudget,
) -> dict[str, Any]:
    fit_idx = site["fit_idx"]
    support_idx = site["support_idx"]
    train_labels = site["train_labels"]
    test_values = site["test_values"]
    test_labels = site["test_labels"]
    clean = site["train_values"]
    injected = site["injected"]

    if arm == "clean_reference":
        fit_values = clean[fit_idx]
        support_values = clean[support_idx]
        drop_census = {
            "n_in": int(fit_values.shape[0]),
            "n_kept": int(fit_values.shape[0]),
            "n_dropped": 0,
            "classes_kept": [
                int(label) for label in sorted(np.unique(train_labels[fit_idx]))
            ],
        }
        workflow = "identity_on_clean"
    elif arm == "injected_identity":
        fit_values, fit_labels_kept, drop_census = drop_nan_rows(
            injected[fit_idx], train_labels[fit_idx]
        )
        support_values, support_labels_kept, support_drop = drop_nan_rows(
            injected[support_idx], train_labels[support_idx]
        )
        model, fit_info = fit_ridge(budget, arm, fit_values, fit_labels_kept)
        delayed = (
            score_model(model, test_values, test_labels)
            if model is not None
            else empty_score(fit_info["reason"], n=int(test_labels.size))
        )
        support = (
            score_model(model, support_values, support_labels_kept)
            if model is not None and support_values.shape[0]
            else empty_score(
                fit_info["reason"] if model is None
                else "NO_USABLE_SUPPORT_ROWS",
                n=int(support_labels_kept.size),
            )
        )
        return {
            "arm": arm,
            "workflow": "identity_drop_nan_training_rows",
            "fit": fit_info,
            "drop_census_fit": drop_census,
            "drop_census_support": support_drop,
            "delayed": delayed,
            "support": support,
        }
    else:
        operator = OPERATOR_FOR_ARM[arm]
        repaired = apply_operator(operator, injected)
        fit_values = repaired[fit_idx]
        support_values = repaired[support_idx]
        drop_census = {
            "n_in": int(fit_values.shape[0]),
            "n_kept": int(fit_values.shape[0]),
            "n_dropped": 0,
            "classes_kept": [
                int(label) for label in sorted(np.unique(train_labels[fit_idx]))
            ],
        }
        workflow = operator

    model, fit_info = fit_ridge(
        budget, arm, fit_values, train_labels[fit_idx]
    )
    delayed = (
        score_model(model, test_values, test_labels)
        if model is not None else empty_score(fit_info["reason"])
    )
    support = (
        score_model(model, support_values, train_labels[support_idx])
        if model is not None else empty_score(fit_info["reason"])
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


def exam(site: dict[str, Any], budget: FitBudget) -> dict[str, Any]:
    return {arm: run_arm(arm=arm, site=site, budget=budget) for arm in ARMS}


def _acc(row: dict[str, Any], surface: str) -> float | None:
    value = row[surface]["accuracy"]
    return None if value is None else float(value)


def _delta(arm_acc: float | None, ref_acc: float | None) -> float | None:
    if arm_acc is None or ref_acc is None:
        return None
    return float(arm_acc - ref_acc)


def rank_key(values: dict[str, float | None]) -> list[str] | None:
    if any(value is None for value in values.values()):
        return None
    return sorted(values, key=lambda name: (-float(values[name]), name))


def judge(arms: dict[str, Any], test_n: int) -> dict[str, Any]:
    step = 1.0 / float(test_n)
    clean_d = _acc(arms["clean_reference"], "delayed")
    ident_d = _acc(arms["injected_identity"], "delayed")
    clean_s = _acc(arms["clean_reference"], "support")
    ident_s = _acc(arms["injected_identity"], "support")
    delayed_delta = {
        arm: _delta(_acc(arms[arm], "delayed"), clean_d) for arm in ARMS
    }
    support_delta = {
        arm: _delta(_acc(arms[arm], "support"), clean_s) for arm in ARMS
    }
    injury = delayed_delta["injected_identity"]
    injury_readable = injury is not None and injury <= INJURY_BAR
    injury_bar_above_floor = abs(INJURY_BAR) >= step

    recoveries: dict[str, Any] = {}
    legal_headroom = False
    best_impute: str | None = None
    best_recovery = None
    if injury is not None and injury < 0.0 and clean_d is not None and ident_d is not None:
        injury_amount = clean_d - ident_d
        for arm in IMPUTE_ARMS:
            impute_d = _acc(arms[arm], "delayed")
            if impute_d is None:
                recoveries[arm] = {"status": "UNSCORED"}
                continue
            recovered = impute_d - ident_d
            fraction = recovered / injury_amount if injury_amount > 0 else None
            recall_drops: dict[str, float | None] = {}
            clean_recalls = arms["clean_reference"]["delayed"]["per_class_recall"]
            impute_recalls = arms[arm]["delayed"]["per_class_recall"]
            recall_ok = True
            for label in clean_recalls:
                clean_r = clean_recalls[label]["recall"]
                impute_r = impute_recalls.get(label, {}).get("recall")
                drop = (
                    None if clean_r is None or impute_r is None
                    else float(clean_r - impute_r)
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
                "delayed_accuracy": impute_d,
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
                best_impute = arm
    else:
        for arm in IMPUTE_ARMS:
            recoveries[arm] = {"status": "INJURY_UNDEFINED"}

    delayed_order = rank_key(delayed_delta)
    support_order = rank_key(support_delta)
    full_order_match = (
        delayed_order is not None
        and support_order is not None
        and delayed_order == support_order
    )
    pair_direction = None
    if best_impute is not None:
        d_ident = delayed_delta["injected_identity"]
        d_best = delayed_delta[best_impute]
        s_ident = support_delta["injected_identity"]
        s_best = support_delta[best_impute]
        if None not in (d_ident, d_best, s_ident, s_best):
            delayed_impute_better = d_best > d_ident
            support_impute_better = s_best > s_ident
            pair_direction = delayed_impute_better == support_impute_better
    b2 = bool(full_order_match or pair_direction is True)

    if any(_acc(arms[arm], "delayed") is None for arm in ARMS):
        verdict = "INSTRUMENT_UNREADABLE"
        reason = (
            "at least one arm could not produce a delayed accuracy; "
            "the B1 injury contrast is undefined"
        )
    elif not injury_readable:
        verdict = "INJURY_NOT_READABLE"
        reason = (
            "clean vs injected+identity delayed Δacc is %s; "
            "pre-registered bar is <= %.3f"
            % (injury, INJURY_BAR)
        )
    elif not legal_headroom:
        verdict = "NO_LEGAL_HEADROOM"
        reason = (
            "injury is readable but no impute arm recovered >=50% "
            "without a class-recall drop > 0.05"
        )
    elif not b2:
        verdict = "SUPPORT_NOT_PREDICTIVE"
        reason = (
            "B1 passed but Support Δacc order does not match delayed "
            "Δacc order, including the identity / best-impute pair"
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
            "injury_bar_above_floor": injury_bar_above_floor,
            "recall_harm_bar_above_floor": RECALL_HARM_BAR >= step,
            "recovery_fraction_bar": RECOVERY_FRACTION_BAR,
        },
        "b1": {
            "injury_delta_acc": injury,
            "injury_readable": injury_readable,
            "injury_bar": INJURY_BAR,
            "legal_headroom": legal_headroom,
            "recoveries": recoveries,
            "best_impute_arm": best_impute,
            "best_recovery_fraction": best_recovery,
        },
        "b2": {
            "delayed_delta_acc": delayed_delta,
            "support_delta_acc": support_delta,
            "delayed_order": delayed_order,
            "support_order": support_order,
            "full_order_match": full_order_match,
            "identity_best_impute_direction_match": pair_direction,
            "passed": b2,
        },
    }


def numeric_fingerprint(arms: dict[str, Any]) -> str:
    compact: dict[str, Any] = {}
    for arm, row in arms.items():
        compact[arm] = {
            "fit": row["fit"],
            "drop_census_fit": row["drop_census_fit"],
            "delayed": row["delayed"],
            "support": row["support"],
        }
    return hashlib.sha256(
        json.dumps(compact, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def out_of_book(site: dict[str, Any], arms: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    length = int(site["train_values"].shape[1])
    p_complete = (1.0 - MISSING_RATE) ** length
    ident = arms["injected_identity"]
    if ident["fit"]["reason"] == "NO_USABLE_TRAINING_ROWS":
        notes.append(
            "per-row 15%% point MCAR on L=%d makes P(complete series)="
            "(0.85)^%d≈%.3e; identity-as-drop-rows therefore kept 0 fit "
            "rows.  This is a structural collision between the "
            "pre-registered identity NaN policy and the pre-registered "
            "injection, not a rate scan."
            % (length, length, p_complete)
        )
    notes.append(
        "CohortHistoryPublicToolGateway requires 2*192 points; GunPoint "
        "L=150 cannot enter that forecast history window.  Missing "
        "signal was read with the same _window_summary coverage / "
        "max-run formulas Fast Agent gates on, treating each held-in "
        "series as one window."
    )
    clean_d = _acc(arms["clean_reference"], "delayed")
    for arm in IMPUTE_ARMS:
        impute_d = _acc(arms[arm], "delayed")
        if clean_d is not None and impute_d is not None:
            notes.append(
                "%s delayed acc=%.6f vs clean %.6f (Δ=%+.6f); reported "
                "as a diagnostic only because identity delayed is %s."
                % (
                    arm,
                    impute_d,
                    clean_d,
                    impute_d - clean_d,
                    "null" if _acc(ident, "delayed") is None else "defined",
                )
            )
    notes.append(
        "evaluation/functional/run_e2_t6_45_frep_a5a3_replay.py remains "
        "untracked; Part 0 allowlist listed the 45-frep artifacts and "
        "the 44-audit runner, not that replay runner."
    )
    return notes


def obligations(
    *,
    site: dict[str, Any],
    official: FitBudget,
    verify: FitBudget,
    two_run: dict[str, Any],
    observation: dict[str, Any],
    clean_observation: dict[str, Any],
    zip_sha_after: str,
    test_sha_after: str,
) -> dict[str, Any]:
    return {
        "llm_calls": 0,
        "agent_invoked": False,
        "rate_scan": False,
        "third_impute": False,
        "preregistered_gates_rewritten": False,
        "fit_budget_used": official.used,
        "fit_budget_cap": FIT_CAP,
        "fit_budget_respected": bool(official.used <= FIT_CAP),
        "fits_by_arm": dict(official.by_arm),
        "verification_fits": verify.used,
        "yahoo_all_reads": 0,
        "noaa_2025_reads": 0,
        "beyond_17520_reads": 0,
        "nab_reads": 0,
        "smd_reads": 0,
        "test_bytes_touched": False,
        "test_sha_unchanged": site["test_sha"] == test_sha_after,
        "zip_bytes_unchanged": site["zip_sha"] == zip_sha_after,
        "loader_output_unmutated": True,
        "injection_after_load": True,
        "injection_ledger_path": (
            (SCRATCH / "injection_ledger.json")
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "missing_signal_after_inject": observation["missing_signal_present"],
        "missing_signal_on_clean": clean_observation["missing_signal_present"],
        "two_run": two_run["status"],
        "flying_files_untouched": [
            "AGENTS.md",
            "README.md",
            "docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md",
            "docs/SUCCESSOR_BRIEF_2026-08-22.md",
        ],
    }


def render_md(payload: dict[str, Any]) -> str:
    judgment = payload["judgment"]
    choice = payload["dataset_choice"]
    arms = payload["arms"]
    obs = payload["observation_injected"]
    clean_obs = payload["observation_clean"]
    lines = [
        "# CLS-1 classification qualification gate",
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
        "- selected: **%s** (rejected %s)"
        % (choice["selected"], choice["rejected"]),
        "- %s" % choice["reason"],
        "- TRAIN n=%d, TEST n=%d, L=%d"
        % (choice["train_n"], choice["test_n"], choice["series_length"]),
        "- TRAIN class counts: %s" % choice["train_class_counts"],
        "- TEST class counts: %s" % choice["test_class_counts"],
        "",
        "## Site",
        "",
        "- Consumer: ridge-raw-plus-difference-v1 "
        "(RidgeClassifier alpha=1, features = raw || first difference; "
        "reused from run_e2_task_context_label_evidence_witness.py)",
        "- quality contract: classification-global-coarse-quality-v1",
        "- held-in = official TRAIN; Query/delayed = official TEST "
        "(byte-zero-touch)",
        "- Support = per-class 30%% of TRAIN, min 3/class, seed %d; "
        "remainder = fit"
        % payload["site"]["support_seed"],
        "- fit n=%d, Support n=%d"
        % (payload["site"]["fit_n"], payload["site"]["support_n"]),
        "- injection: held-in only, %s, rate %.2f, seed %d, "
        "%d missing points/row"
        % (
            payload["site"]["missing_form"],
            payload["site"]["missing_rate"],
            payload["site"]["inject_seed"],
            payload["site"]["n_missing_per_row"],
        ),
        "- ledger: `%s`" % payload["site"]["ledger_path"],
        "",
        "## Observation missing signal",
        "",
        "| surface | coverage | max_missing_run | missing_signal |",
        "|---|---:|---:|---|",
        "| clean TRAIN | %.6f | %d | %s |"
        % (
            clean_obs["recent.coverage"],
            clean_obs["recent.maximum_missing_run_length"],
            clean_obs["missing_signal_present"],
        ),
        "| injected held-in | %.6f | %d | %s |"
        % (
            obs["recent.coverage"],
            obs["recent.maximum_missing_run_length"],
            obs["missing_signal_present"],
        ),
        "",
        "- Fast `_MISSING_ONLY_OPS` would skip impute: **%s** "
        "(need coverage<1 or max_run>0)"
        % obs["impute_ops_would_be_skipped"],
        "- mean per-series public `missing_fraction`: %.6f"
        % obs["mean_missing_fraction"],
        "",
        "## Four-arm delayed (TEST) and Support",
        "",
        "| arm | workflow | n_fit (dropped) | delayed acc | Support acc | "
        "delayed Δacc | Support Δacc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
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
    lines.extend([
        "",
        "### Per-class recall (delayed / Support)",
        "",
    ])
    for arm in ARMS:
        row = arms[arm]
        lines.append("**%s**" % arm)
        lines.append("")
        lines.append("| class | delayed n | delayed recall | Support n | "
                     "Support recall |")
        lines.append("|---|---:|---:|---:|---:|")
        delayed_r = row["delayed"]["per_class_recall"]
        support_r = row["support"]["per_class_recall"]
        labels = sorted(set(delayed_r) | set(support_r), key=int)
        if not labels:
            lines.append("| — | 0 | null | 0 | null |")
        for label in labels:
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
        "- injury Δacc (identity − clean, delayed): **%s** "
        "(readable=%s, bar=%.3f, bar/floor=%.2f steps)"
        % (
            "null" if judgment["b1"]["injury_delta_acc"] is None
            else "%+.6f" % judgment["b1"]["injury_delta_acc"],
            judgment["b1"]["injury_readable"],
            judgment["b1"]["injury_bar"],
            judgment["quantization"]["injury_bar_abs_steps"],
        ),
        "- legal headroom: **%s**" % judgment["b1"]["legal_headroom"],
        "- best impute: %s (recovery fraction %s)"
        % (
            judgment["b1"]["best_impute_arm"],
            "null" if judgment["b1"]["best_recovery_fraction"] is None
            else "%.4f" % judgment["b1"]["best_recovery_fraction"],
        ),
        "- Support vs delayed full order match: %s"
        % judgment["b2"]["full_order_match"],
        "- identity / best-impute direction match: %s"
        % judgment["b2"]["identity_best_impute_direction_match"],
        "- B2 passed: **%s**" % judgment["b2"]["passed"],
        "",
        "### Impute recovery detail",
        "",
        "| arm | recovery fraction | recall guard | qualifies |",
        "|---|---:|---|---|",
    ])
    for arm in IMPUTE_ARMS:
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
    det = payload["determinism"]
    lines.extend([
        "",
        "## n and quantization floor",
        "",
        "- TEST n=%d, one step=1/n=%.6f" % (q["test_n"], q["step"]),
        "- injury bar 0.05 is %.2f steps (above floor=%s)"
        % (q["injury_bar_abs_steps"], q["injury_bar_above_floor"]),
        "- recall-harm bar 0.05 above floor: %s"
        % q["recall_harm_bar_above_floor"],
        "",
        "## Fit ledger and determinism",
        "",
        "- official fits %d / %d: %s"
        % (cost["fits"], cost["fit_cap"], cost["fits_by_arm"]),
        "- verification fits %d (reported separately)"
        % cost["verification_fits"],
        "- two-run numeric fingerprint: **%s**" % det["two_run"],
        "- TEST SHA unchanged: %s" % det["test_sha_unchanged"],
        "- zip SHA unchanged: %s" % det["zip_sha_unchanged"],
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
    task_context = classification_task_context_v1(
        task_spec=classification_task_spec_v1(
            downstream_model_class="ridge-raw-plus-difference-v1"
        ),
        quality_contract=classification_global_coarse_task_quality_contract_v1(),
    )
    site = load_site()
    clean_obs = missing_observation(site["train_values"])
    injected_obs = missing_observation(site["injected"])
    if clean_obs["missing_signal_present"]:
        raise Stop(
            "INSTRUMENT_UNREADABLE",
            "clean TRAIN already shows a missing signal; the loader/site "
            "is not a clean reference",
        )
    if not injected_obs["missing_signal_present"]:
        raise Stop(
            "INSTRUMENT_UNREADABLE",
            "injected held-in has coverage=1 and max_missing_run=0; "
            "Fast Agent would skip every impute operator",
        )

    official = FitBudget(FIT_CAP)
    arms = exam(site, official)
    verify = FitBudget(FIT_CAP)
    again = exam(site, verify)
    fp1 = numeric_fingerprint(arms)
    fp2 = numeric_fingerprint(again)
    two_run = {
        "status": "BITWISE_IDENTICAL" if fp1 == fp2 else "DRIFT",
        "official_fingerprint": fp1,
        "verification_fingerprint": fp2,
    }
    if fp1 != fp2:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "two-run numeric fingerprint drifted")

    test_sha_after = _array_sha(site["test_values"])
    zip_sha_after = _file_sha(site["archive"])
    if test_sha_after != site["test_sha"]:
        raise Stop("PROTOCOL_BREACH", "TEST array bytes changed")
    if zip_sha_after != site["zip_sha"]:
        raise Stop("PROTOCOL_BREACH", "UCR zip bytes changed")

    judgment = judge(arms, int(site["test_labels"].size))
    choice = dataset_choice_reason(site)
    support_counts = {
        str(label): int(np.count_nonzero(
            site["train_labels"][site["support_idx"]] == label
        ))
        for label in sorted(int(value) for value in np.unique(site["train_labels"]))
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "book": "CLS-1 classification qualification gate",
        "evidence_class": "INSTRUMENT / POSITIVE_CONTROL",
        "development_only": True,
        "claim_cap": (
            "development positive control on an injected MCAR defect; "
            "not a natural UCR capability claim"
        ),
        "task_context": task_context.to_dict(),
        "dataset_choice": choice,
        "site": {
            "dataset": DATASET,
            "archive": "%s/%s.zip" % (DATA_DIR, DATASET),
            "held_in": "official TRAIN (only injectable / processable region)",
            "query": "official TEST (delayed scoring; byte-zero-touch)",
            "support_fraction": SUPPORT_FRACTION,
            "support_seed": SEED_SUPPORT,
            "inject_seed": SEED_INJECT,
            "missing_form": MISSING_FORM,
            "missing_rate": MISSING_RATE,
            "n_missing_per_row": site["ledger"]["n_missing_per_row"],
            "fit_n": int(site["fit_idx"].size),
            "support_n": int(site["support_idx"].size),
            "support_class_counts": support_counts,
            "fit_idx": [int(index) for index in site["fit_idx"]],
            "support_idx": [int(index) for index in site["support_idx"]],
            "ledger_path": (
                (SCRATCH / "injection_ledger.json")
                .relative_to(PROJECT_ROOT)
                .as_posix()
            ),
            "identity_nan_policy": (
                "drop training rows that contain any NaN; honest "
                "no-treatment equals lost row count"
            ),
        },
        "observation_clean": {
            key: clean_obs[key]
            for key in (
                "recent.coverage",
                "recent.maximum_missing_run_length",
                "missing_run_count",
                "missing_signal_present",
                "impute_ops_would_be_skipped",
                "mean_missing_fraction",
            )
        },
        "observation_injected": {
            key: injected_obs[key]
            for key in (
                "recent.coverage",
                "recent.maximum_missing_run_length",
                "missing_run_count",
                "missing_signal_present",
                "fast_agent_missing_only_ops",
                "impute_ops_would_be_skipped",
                "mean_missing_fraction",
            )
        },
        "arms": arms,
        "judgment": judgment,
        "determinism": {
            "two_run": two_run["status"],
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
        "obligations": obligations(
            site=site,
            official=official,
            verify=verify,
            two_run=two_run,
            observation=injected_obs,
            clean_observation=clean_obs,
            zip_sha_after=zip_sha_after,
            test_sha_after=test_sha_after,
        ),
        "out_of_book": out_of_book(site, arms),
    }
    # keep the per-series public-feature dump in JSON only
    payload["observation_injected"]["per_series_count"] = len(
        injected_obs["per_series_public_features"]
    )
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(_json_text(payload), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": judgment["verdict"],
        "reason": judgment["reason"],
        "dataset": DATASET,
        "delayed": {
            arm: arms[arm]["delayed"]["accuracy"] for arm in ARMS
        },
        "support": {
            arm: arms[arm]["support"]["accuracy"] for arm in ARMS
        },
        "observation": {
            "coverage": injected_obs["recent.coverage"],
            "max_run": injected_obs["recent.maximum_missing_run_length"],
            "missing_signal": injected_obs["missing_signal_present"],
        },
        "fits": official.used,
        "two_run": two_run["status"],
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
    except Stop as exc:
        print(json.dumps({
            "verdict": exc.code,
            "detail": exc.detail,
        }, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
