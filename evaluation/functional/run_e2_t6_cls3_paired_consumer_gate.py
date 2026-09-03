"""CLS-3 -- paired Consumer qualification gate (sol-frozen).

Same GunPoint development substrate and the frozen CLS-2 contiguous-burst
ledger.  Two Consumers, same features (raw || first difference):

  ridge-raw-plus-difference-v1  (RidgeClassifier alpha=1)
  knn-k3-raw-plus-difference-v1 (KNeighborsClassifier k=3, Euclidean, uniform)

Workflows frozen: identity + hampel_filter + outlier_mad.  No extras.
k / distance / amplitude / dataset are not scanned.  0 LLM.

This is a paired-Consumer eligibility reading, not a judge swap to harvest
a positive CLS-2 result.

Usage:
  python evaluation/functional/run_e2_t6_cls3_paired_consumer_gate.py --run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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

import run_e2_t6_cls1_qualification_gate as cls1  # noqa: E402
import run_e2_t6_cls2_value_corruption_gate as cls2  # noqa: E402
import run_e2_task_context_label_evidence_witness as witness  # noqa: E402

PROTOCOL_VERSION = "t6_cls3_paired_consumer_gate_v1"
RUN_ID = "cls3_v1"
DATASET = "GunPoint"
LEDGER_PATH = PROJECT_ROOT / "_scratch" / "cls2" / "cls2_v1" / "injection_ledger.json"
CLS2_INJECTED_SHA = "2e1b1049f6729610eb56cd8240d486803455baa34d7a358939e3f13d0aa7f45e"
CLS2_TEST_SHA = "722f7f7cb2eb99b65d2276d1c0b394fb929e859a75d5f452de7b28f1e554ad01"
CLS2_ZIP_SHA = "d7513cfe222418fabfdb5a6434ffb21ac3de4923e637971e9388ebc857816803"

CONSUMERS = ("ridge", "knn")
WORKFLOWS = (
    "clean_reference",
    "corrupted_identity",
    "corrupted_hampel",
    "corrupted_outlier_mad",
)
REPAIR_WORKFLOWS = ("corrupted_hampel", "corrupted_outlier_mad")
OPERATOR_FOR_WORKFLOW = {
    "corrupted_hampel": "hampel_filter",
    "corrupted_outlier_mad": "outlier_mad",
}
IDENTITY_WORKFLOW = "corrupted_identity"

KNN_K = 3
KNN_METRIC = "euclidean"
KNN_WEIGHTS = "uniform"
FIT_CAP = 30
INJURY_BAR = cls1.INJURY_BAR
RECOVERY_FRACTION_BAR = cls1.RECOVERY_FRACTION_BAR
RECALL_HARM_BAR = cls1.RECALL_HARM_BAR
SEED_SUPPORT = cls1.SEED_SUPPORT

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_cls3_paired_consumer_gate.json"
OUT_MD = E2 / "t6_cls3_paired_consumer_gate.md"


def arm_key(consumer: str, workflow: str) -> str:
    return "%s.%s" % (consumer, workflow)


def apply_frozen_ledger(
    values: np.ndarray,
    ledger: dict[str, Any],
) -> np.ndarray:
    injected = np.asarray(values, dtype=np.float64).copy()
    n_rows, length = injected.shape
    if int(ledger["n_rows"]) != n_rows or int(ledger["series_length"]) != length:
        raise cls1.Stop(
            "INSTRUMENT_UNREADABLE",
            "ledger shape %sx%s does not match TRAIN %sx%s"
            % (ledger["n_rows"], ledger["series_length"], n_rows, length),
        )
    if ledger.get("form") != "row_subset_one_contiguous_additive_burst":
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "ledger form is not the CLS-2 burst")
    if int(ledger.get("seed", -1)) != cls2.SEED_INJECT:
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "ledger seed is not CLS-2 SEED_INJECT")
    hit = {int(row) for row in ledger["hit_rows"]}
    seen: set[int] = set()
    for record in ledger["rows"]:
        row = int(record["row"])
        start = int(record["start"])
        seg_len = int(record["length"])
        noise = np.asarray(record["noise"], dtype=np.float64)
        if row not in hit:
            raise cls1.Stop("INSTRUMENT_UNREADABLE", "ledger row %d not in hit_rows" % row)
        if noise.size != seg_len:
            raise cls1.Stop(
                "INSTRUMENT_UNREADABLE",
                "ledger row %d noise length %d != segment %d" % (row, noise.size, seg_len),
            )
        if start < 0 or start + seg_len > length:
            raise cls1.Stop("INSTRUMENT_UNREADABLE", "ledger row %d segment out of range" % row)
        injected[row, start:start + seg_len] = values[row, start:start + seg_len] + noise
        seen.add(row)
    if seen != hit:
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "ledger rows do not cover hit_rows")
    for row in range(n_rows):
        if row in hit:
            continue
        if not np.array_equal(injected[row], values[row]):
            raise cls1.Stop("PROTOCOL_BREACH", "clean row %d was mutated" % row)
    if not np.isfinite(injected).all():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "ledger replay produced non-finite values")
    if np.array_equal(injected, values):
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "ledger replay left TRAIN unchanged")
    return injected


def fit_knn(
    budget: cls1.FitBudget,
    arm: str,
    train_values: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    n_rows = int(train_values.shape[0])
    classes = sorted(int(label) for label in np.unique(train_labels)) if n_rows else []
    if n_rows < 2 or len(classes) < 2:
        return None, {
            "fit": False,
            "reason": "NO_USABLE_TRAINING_ROWS",
            "n_train": n_rows,
            "classes": classes,
            "consumer": "knn",
            "knn_k": KNN_K,
        }
    if not np.isfinite(train_values).all():
        return None, {
            "fit": False,
            "reason": "NON_FINITE_TRAINING_ROWS",
            "n_train": n_rows,
            "classes": classes,
            "consumer": "knn",
            "knn_k": KNN_K,
        }
    budget.spend(arm)
    model = KNeighborsClassifier(
        n_neighbors=KNN_K,
        metric=KNN_METRIC,
        weights=KNN_WEIGHTS,
    )
    model.fit(cls1._features(train_values), train_labels)
    return model, {
        "fit": True,
        "reason": None,
        "n_train": n_rows,
        "classes": classes,
        "consumer": "knn",
        "knn_k": KNN_K,
        "knn_metric": KNN_METRIC,
        "knn_weights": KNN_WEIGHTS,
    }


def fit_consumer(
    consumer: str,
    budget: cls1.FitBudget,
    arm: str,
    train_values: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    if consumer == "ridge":
        model, info = cls1.fit_ridge(budget, arm, train_values, train_labels)
        info = dict(info)
        info["consumer"] = "ridge"
        return model, info
    if consumer == "knn":
        return fit_knn(budget, arm, train_values, train_labels)
    raise cls1.Stop("INSTRUMENT_UNREADABLE", "unknown consumer %s" % consumer)


def workflow_matrices(site: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    fit_idx = site["fit_idx"]
    support_idx = site["support_idx"]
    matrices: dict[str, dict[str, np.ndarray]] = {
        "clean_reference": {
            "fit": site["train_values"][fit_idx],
            "support": site["train_values"][support_idx],
        },
        IDENTITY_WORKFLOW: {
            "fit": site["injected"][fit_idx],
            "support": site["injected"][support_idx],
        },
    }
    for workflow, operator in OPERATOR_FOR_WORKFLOW.items():
        repaired = cls1.apply_operator(operator, site["injected"])
        matrices[workflow] = {
            "fit": repaired[fit_idx],
            "support": repaired[support_idx],
        }
    return matrices


def run_arm(
    *,
    consumer: str,
    workflow: str,
    site: dict[str, Any],
    matrices: dict[str, dict[str, np.ndarray]],
    budget: cls1.FitBudget,
) -> dict[str, Any]:
    labels = site["train_labels"]
    fit_idx = site["fit_idx"]
    support_idx = site["support_idx"]
    block = matrices[workflow]
    if workflow == "clean_reference":
        name = "identity_on_clean"
    elif workflow == IDENTITY_WORKFLOW:
        name = "identity_on_corrupted"
    else:
        name = OPERATOR_FOR_WORKFLOW[workflow]
    drop_census = {
        "n_in": int(block["fit"].shape[0]),
        "n_kept": int(block["fit"].shape[0]),
        "n_dropped": 0,
        "classes_kept": [
            int(label) for label in sorted(np.unique(labels[fit_idx]))
        ],
        "note": "value corruption stays finite; identity does not drop rows",
    }
    key = arm_key(consumer, workflow)
    model, fit_info = fit_consumer(
        consumer, budget, key, block["fit"], labels[fit_idx]
    )
    delayed = (
        cls1.score_model(model, site["test_values"], site["test_labels"])
        if model is not None
        else cls1.empty_score(fit_info["reason"], n=int(site["test_labels"].size))
    )
    support = (
        cls1.score_model(model, block["support"], labels[support_idx])
        if model is not None
        else cls1.empty_score(fit_info["reason"], n=int(support_idx.size))
    )
    return {
        "arm": key,
        "consumer": consumer,
        "workflow": name,
        "workflow_key": workflow,
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
    matrices = workflow_matrices(site)
    arms: dict[str, Any] = {}
    for consumer in CONSUMERS:
        for workflow in WORKFLOWS:
            arms[arm_key(consumer, workflow)] = run_arm(
                consumer=consumer,
                workflow=workflow,
                site=site,
                matrices=matrices,
                budget=budget,
            )
    return arms


def consumer_slice(arms: dict[str, Any], consumer: str) -> dict[str, Any]:
    return {workflow: arms[arm_key(consumer, workflow)] for workflow in WORKFLOWS}


def recoveries_for(
    slice_arms: dict[str, Any],
    clean_d: float | None,
    ident_d: float | None,
    injury: float | None,
) -> tuple[dict[str, Any], bool, str | None, float | None]:
    recoveries: dict[str, Any] = {}
    legal = False
    best_repair: str | None = None
    best_recovery: float | None = None
    if injury is None or injury >= 0.0 or clean_d is None or ident_d is None:
        for workflow in REPAIR_WORKFLOWS:
            recoveries[workflow] = {"status": "INJURY_UNDEFINED"}
        return recoveries, False, None, None
    injury_amount = clean_d - ident_d
    for workflow in REPAIR_WORKFLOWS:
        repair_d = cls1._acc(slice_arms[workflow], "delayed")
        if repair_d is None:
            recoveries[workflow] = {"status": "UNSCORED"}
            continue
        recovered = repair_d - ident_d
        fraction = recovered / injury_amount if injury_amount > 0 else None
        clean_recalls = slice_arms["clean_reference"]["delayed"]["per_class_recall"]
        repair_recalls = slice_arms[workflow]["delayed"]["per_class_recall"]
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
        recovery_ok = fraction is not None and fraction >= RECOVERY_FRACTION_BAR
        qualifies = bool(recovery_ok and recall_ok)
        recoveries[workflow] = {
            "delayed_accuracy": repair_d,
            "recovered_acc": recovered,
            "recovery_fraction": fraction,
            "recovery_ge_50": recovery_ok,
            "recall_drop_vs_clean": recall_drops,
            "recall_guard_ok": recall_ok,
            "qualifies": qualifies,
        }
        if qualifies:
            legal = True
        if fraction is not None and (best_recovery is None or fraction > best_recovery):
            best_recovery = fraction
            best_repair = workflow
    return recoveries, legal, best_repair, best_recovery


def judge(arms: dict[str, Any], test_n: int) -> dict[str, Any]:
    step = 1.0 / float(test_n)
    per_consumer: dict[str, Any] = {}
    for consumer in CONSUMERS:
        slice_arms = consumer_slice(arms, consumer)
        clean_d = cls1._acc(slice_arms["clean_reference"], "delayed")
        ident_d = cls1._acc(slice_arms[IDENTITY_WORKFLOW], "delayed")
        clean_s = cls1._acc(slice_arms["clean_reference"], "support")
        delayed_delta = {
            workflow: cls1._delta(cls1._acc(slice_arms[workflow], "delayed"), clean_d)
            for workflow in WORKFLOWS
        }
        support_delta = {
            workflow: cls1._delta(cls1._acc(slice_arms[workflow], "support"), clean_s)
            for workflow in WORKFLOWS
        }
        injury = delayed_delta[IDENTITY_WORKFLOW]
        recoveries, legal, best_repair, best_recovery = recoveries_for(
            slice_arms, clean_d, ident_d, injury
        )
        delayed_order = cls1.rank_key(delayed_delta)
        support_order = cls1.rank_key(support_delta)
        full_order_match = (
            delayed_order is not None
            and support_order is not None
            and delayed_order == support_order
        )
        pair_direction = None
        if best_repair is not None:
            d_ident = delayed_delta[IDENTITY_WORKFLOW]
            d_best = delayed_delta[best_repair]
            s_ident = support_delta[IDENTITY_WORKFLOW]
            s_best = support_delta[best_repair]
            if None not in (d_ident, d_best, s_ident, s_best):
                pair_direction = (d_best > d_ident) == (s_best > s_ident)
        per_consumer[consumer] = {
            "clean_delayed_acc": clean_d,
            "identity_delayed_acc": ident_d,
            "injury_delta_acc": injury,
            "injury_readable": injury is not None and injury <= INJURY_BAR,
            "legal_headroom": legal,
            "recoveries": recoveries,
            "best_repair_workflow": best_repair,
            "best_recovery_fraction": best_recovery,
            "delayed_delta_acc": delayed_delta,
            "support_delta_acc": support_delta,
            "delayed_order": delayed_order,
            "support_order": support_order,
            "full_order_match": full_order_match,
            "identity_best_repair_direction_match": pair_direction,
        }

    knn = per_consumer["knn"]
    ridge = per_consumer["ridge"]
    any_unscored = any(
        cls1._acc(arms[arm_key(consumer, workflow)], "delayed") is None
        for consumer in CONSUMERS
        for workflow in WORKFLOWS
    )
    knn_recovery_ge_50 = any(
        item.get("recovery_ge_50") is True
        for item in knn["recoveries"].values()
    )
    knn_recall_ok_on_ge50 = any(
        item.get("recovery_ge_50") is True and item.get("recall_guard_ok") is True
        for item in knn["recoveries"].values()
    )
    conditions = {
        "knn_injury_readable": bool(knn["injury_readable"]),
        "knn_recovery_ge_50": bool(knn_recovery_ge_50),
        "knn_recall_guard_on_ge50_repair": bool(knn_recall_ok_on_ge50),
        "knn_support_predicts_delayed_order": bool(knn["full_order_match"]),
        "ridge_identity_remains_numb": bool(
            ridge["injury_delta_acc"] is not None
            and ridge["injury_delta_acc"] > INJURY_BAR
        ),
    }
    all_five = all(conditions.values())

    if any_unscored:
        verdict = "INSTRUMENT_UNREADABLE"
        reason = "at least one of the eight arms could not produce a delayed accuracy"
    elif not conditions["knn_injury_readable"]:
        verdict = "KNN_ALSO_NUMB"
        reason = (
            "kNN clean vs corrupted+identity delayed Δacc is %s; "
            "material bar is <= %.3f; contiguous-burst family closes"
            % (knn["injury_delta_acc"], INJURY_BAR)
        )
    elif not knn["legal_headroom"]:
        verdict = "KNN_INJURED_NO_REPAIR"
        reason = (
            "kNN injury is readable but no frozen Workflow recovered >=50% "
            "without a class-recall drop > 0.05 (Program Supply gap; no Agent)"
        )
    elif not conditions["knn_support_predicts_delayed_order"]:
        verdict = "SUPPORT_NOT_PREDICTIVE"
        reason = (
            "kNN Support Δacc arm order does not match delayed Δacc arm order "
            "(Feedback first-fault; no A5/A3)"
        )
    elif not conditions["ridge_identity_remains_numb"]:
        verdict = "INSTRUMENT_UNREADABLE"
        reason = (
            "kNN is injured and repaired with Support-predictive order, but "
            "ridge corrupted+identity Δacc is %s (not > %.3f); Consumer "
            "contrast is missing"
            % (ridge["injury_delta_acc"], INJURY_BAR)
        )
    elif all_five:
        verdict = "PAIRED_HEADROOM_CONFIRMED"
        reason = (
            "kNN is injured and legally repaired, Support order matches delayed, "
            "and ridge stays numb: paired-Consumer eligibility holds"
        )
    else:
        verdict = "INSTRUMENT_UNREADABLE"
        reason = "five-condition table did not map onto a named exit"

    return {
        "verdict": verdict,
        "reason": reason,
        "role": (
            "paired Consumer qualification; not a judge swap on the CLS-2 ridge result"
        ),
        "quantization": {
            "test_n": int(test_n),
            "step": step,
            "injury_bar": INJURY_BAR,
            "injury_bar_abs_steps": abs(INJURY_BAR) / step,
            "injury_bar_above_floor": abs(INJURY_BAR) >= step,
            "recall_harm_bar_above_floor": RECALL_HARM_BAR >= step,
            "recovery_fraction_bar": RECOVERY_FRACTION_BAR,
        },
        "five_conditions": conditions,
        "all_five": all_five,
        "per_consumer": per_consumer,
        "consumer_contrast": {
            "knn_injury_delta_acc": knn["injury_delta_acc"],
            "ridge_injury_delta_acc": ridge["injury_delta_acc"],
            "delta_injury_knn_minus_ridge": (
                None
                if knn["injury_delta_acc"] is None or ridge["injury_delta_acc"] is None
                else float(knn["injury_delta_acc"] - ridge["injury_delta_acc"])
            ),
            "ridge_stays_above_bar": conditions["ridge_identity_remains_numb"],
        },
    }


def load_site() -> dict[str, Any]:
    if not LEDGER_PATH.is_file():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "missing CLS-2 ledger %s" % LEDGER_PATH)
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    archive = PROJECT_ROOT / witness.DATA_DIR / ("%s.zip" % DATASET)
    if not archive.is_file():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "missing archive %s" % archive)
    zip_sha_before = cls1._file_sha(archive)
    if zip_sha_before != CLS2_ZIP_SHA:
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "GunPoint zip SHA drifted from CLS-2")
    train_values, train_labels = witness._load_split(np, archive, DATASET, "TRAIN")
    test_values, test_labels = witness._load_split(np, archive, DATASET, "TEST")
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "loader emitted non-finite values")
    fit_idx, support_idx = cls1.split_fit_support(train_labels, SEED_SUPPORT)
    injected = apply_frozen_ledger(train_values, ledger)
    injected_again = apply_frozen_ledger(train_values, ledger)
    if cls1._array_sha(injected) != cls1._array_sha(injected_again):
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "ledger replay drifted")
    injected_sha = cls1._array_sha(injected)
    if injected_sha != CLS2_INJECTED_SHA:
        raise cls1.Stop(
            "INSTRUMENT_UNREADABLE",
            "ledger replay SHA %s != CLS-2 injected SHA %s"
            % (injected_sha, CLS2_INJECTED_SHA),
        )
    reload_train, _labels = witness._load_split(np, archive, DATASET, "TRAIN")
    if not np.array_equal(train_values, reload_train):
        raise cls1.Stop("PROTOCOL_BREACH", "TRAIN memory copy drifted after ledger replay")
    test_sha = cls1._array_sha(test_values)
    if test_sha != CLS2_TEST_SHA:
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "TEST SHA drifted from CLS-2")
    if cls1._file_sha(archive) != zip_sha_before:
        raise cls1.Stop("PROTOCOL_BREACH", "UCR zip bytes changed during site build")
    ledger_after = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    if cls1._json_text(ledger) != cls1._json_text(ledger_after):
        raise cls1.Stop("PROTOCOL_BREACH", "CLS-2 ledger bytes changed during site build")
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
        "train_sha": cls1._array_sha(train_values),
        "injected_sha": injected_sha,
        "ledger_sha": cls1._file_sha(LEDGER_PATH),
    }


def out_of_book(site: dict[str, Any], arms: dict[str, Any], judgment: dict[str, Any]) -> list[str]:
    notes = [
        "Paired-Consumer eligibility, not a judge swap to overturn CLS-2 INJURY_NOT_READABLE.",
        "Injection was ledger replay only; seed/segments/noise were not redrawn.",
        "k=3 / Euclidean / uniform and GunPoint / amplitude 5 were not scanned.",
        "Part 0 is empty; CLS-2 artifacts stay with the CLS-OP book.",
        "This CLS-3 artifact stays uncommitted.",
    ]
    ridge = judgment["per_consumer"]["ridge"]
    cls2_ridge = {
        "clean_reference": 0.82,
        "corrupted_identity": 0.8066666666666666,
        "corrupted_hampel": 0.8066666666666666,
        "corrupted_outlier_mad": 0.6933333333333334,
    }
    drifted = []
    for workflow, expected in cls2_ridge.items():
        got = cls1._acc(arms[arm_key("ridge", workflow)], "delayed")
        if got is None or abs(got - expected) > 1e-12:
            drifted.append("%s got %s expected %s" % (workflow, got, expected))
    if drifted:
        notes.append("Ridge delayed acc drifted from CLS-2: %s." % "; ".join(drifted))
    else:
        notes.append("Ridge four-arm delayed acc reproduced CLS-2 bitwise.")
    knn_injury = judgment["per_consumer"]["knn"]["injury_delta_acc"]
    if knn_injury is not None:
        notes.append(
            "kNN identity delayed Δacc=%+.6f (bar=−0.05, step=%.6f)."
            % (knn_injury, 1.0 / float(site["test_labels"].size))
        )
    if not judgment["per_consumer"]["knn"]["full_order_match"]:
        notes.append(
            "kNN Support vs delayed full order failed "
            "(delayed %s; Support %s; pair-direction %s)."
            % (
                judgment["per_consumer"]["knn"]["delayed_order"],
                judgment["per_consumer"]["knn"]["support_order"],
                judgment["per_consumer"]["knn"]["identity_best_repair_direction_match"],
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
    lines = [
        "# CLS-3 paired Consumer qualification gate",
        "",
        "evidence class: %s (development).  %s"
        % (payload["evidence_class"], payload["claim_cap"]),
        "",
        "## Verdict",
        "",
        "- **%s**" % judgment["verdict"],
        "- %s" % judgment["reason"],
        "- role: %s" % judgment["role"],
        "",
        "## Five conditions",
        "",
    ]
    labels = {
        "knn_injury_readable": "(1) kNN delayed injury ≤ −0.05",
        "knn_recovery_ge_50": "(2) ≥1 Workflow recovers ≥50%",
        "knn_recall_guard_on_ge50_repair": "(3) that repair does not drop class recall >0.05",
        "knn_support_predicts_delayed_order": "(4) kNN Support order matches delayed",
        "ridge_identity_remains_numb": "(5) ridge identity Δacc still > −0.05",
    }
    for key, title in labels.items():
        lines.append("- %s: **%s**" % (title, judgment["five_conditions"][key]))
    lines.extend([
        "- all five: **%s**" % judgment["all_five"],
        "",
        "## Consumer contrast",
        "",
        "- kNN identity Δacc: **%s**"
        % _fmt_delta(judgment["consumer_contrast"]["knn_injury_delta_acc"]),
        "- ridge identity Δacc: **%s**"
        % _fmt_delta(judgment["consumer_contrast"]["ridge_injury_delta_acc"]),
        "- kNN − ridge injury: %s"
        % _fmt_delta(judgment["consumer_contrast"]["delta_injury_knn_minus_ridge"]),
        "",
        "## Site",
        "",
        "- dataset: GunPoint development (no ladder)",
        "- injection: CLS-2 ledger replay (`%s`)" % payload["site"]["ledger_path"],
        "- seed/segments/amplitude: frozen; zero redraw",
        "- features: raw || first difference (shared)",
        "- Consumers: ridge-raw-plus-difference-v1; knn k=3 Euclidean uniform",
        "- Workflows: identity + hampel_filter + outlier_mad (no extras)",
        "- held-in = official TRAIN; TEST = delayed, byte-zero-touch",
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
        "## Eight-arm delayed (TEST) and Support",
        "",
        "| consumer | workflow | n_fit | delayed acc | Support acc | "
        "delayed Δacc | Support Δacc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for consumer in CONSUMERS:
        deltas = judgment["per_consumer"][consumer]
        for workflow in WORKFLOWS:
            row = arms[arm_key(consumer, workflow)]
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s |"
                % (
                    consumer,
                    row["workflow"],
                    row["fit"]["n_train"],
                    _fmt_acc(row["delayed"]["accuracy"]),
                    _fmt_acc(row["support"]["accuracy"]),
                    _fmt_delta(deltas["delayed_delta_acc"][workflow]),
                    _fmt_delta(deltas["support_delta_acc"][workflow]),
                )
            )
    lines.extend(["", "### Per-class recall (delayed / Support)", ""])
    for consumer in CONSUMERS:
        for workflow in WORKFLOWS:
            row = arms[arm_key(consumer, workflow)]
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
    lines.extend(["## kNN recoveries", ""])
    knn_rec = judgment["per_consumer"]["knn"]["recoveries"]
    lines.append("| workflow | recovery fraction | recall guard | qualifies |")
    lines.append("|---|---:|---|---|")
    for workflow in REPAIR_WORKFLOWS:
        item = knn_rec.get(workflow, {})
        if "qualifies" not in item:
            lines.append("| %s | — | — | %s |" % (workflow, item.get("status")))
            continue
        frac = item["recovery_fraction"]
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                workflow,
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
    ridge_ctx = classification_task_context_v1(
        task_spec=classification_task_spec_v1(
            downstream_model_class="ridge-raw-plus-difference-v1"
        ),
        quality_contract=classification_global_coarse_task_quality_contract_v1(),
    )
    knn_ctx = classification_task_context_v1(
        task_spec=classification_task_spec_v1(
            downstream_model_class="knn-k3-raw-plus-difference-v1"
        ),
        quality_contract=classification_global_coarse_task_quality_contract_v1(),
    )
    site = load_site()
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
    ledger_sha_after = cls1._file_sha(LEDGER_PATH)
    if test_sha_after != site["test_sha"]:
        raise cls1.Stop("PROTOCOL_BREACH", "TEST array bytes changed")
    if zip_sha_after != site["zip_sha"]:
        raise cls1.Stop("PROTOCOL_BREACH", "UCR zip bytes changed")
    if ledger_sha_after != site["ledger_sha"]:
        raise cls1.Stop("PROTOCOL_BREACH", "CLS-2 ledger bytes changed")
    judgment = judge(arms, int(site["test_labels"].size))
    hit = set(int(row) for row in site["ledger"]["hit_rows"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "book": "CLS-3 paired Consumer qualification gate",
        "evidence_class": "INSTRUMENT / POSITIVE_CONTROL",
        "development_only": True,
        "claim_cap": (
            "development paired-Consumer eligibility on the frozen CLS-2 burst; "
            "not a natural UCR capability claim and not a CLS-2 judge swap"
        ),
        "task_context": {
            "ridge": ridge_ctx.to_dict(),
            "knn": knn_ctx.to_dict(),
        },
        "consumers": {
            "ridge": {
                "id": "ridge-raw-plus-difference-v1",
                "estimator": "sklearn.linear_model.RidgeClassifier",
                "alpha": witness.RIDGE_ALPHA,
                "features": "raw || first difference",
            },
            "knn": {
                "id": "knn-k3-raw-plus-difference-v1",
                "estimator": "sklearn.neighbors.KNeighborsClassifier",
                "n_neighbors": KNN_K,
                "metric": KNN_METRIC,
                "weights": KNN_WEIGHTS,
                "features": "raw || first difference",
            },
        },
        "workflows": ["identity", "hampel_filter", "outlier_mad"],
        "site": {
            "dataset": DATASET,
            "archive": "%s/%s.zip" % (witness.DATA_DIR, DATASET),
            "inject_seed": cls2.SEED_INJECT,
            "support_seed": SEED_SUPPORT,
            "train_n": int(site["train_labels"].size),
            "test_n": int(site["test_labels"].size),
            "series_length": int(site["train_values"].shape[1]),
            "fit_n": int(site["fit_idx"].size),
            "support_n": int(site["support_idx"].size),
            "n_hit_rows": site["ledger"]["n_hit_rows"],
            "fit_rows_hit": int(sum(int(i) in hit for i in site["fit_idx"])),
            "support_rows_hit": int(sum(int(i) in hit for i in site["support_idx"])),
            "ledger_path": LEDGER_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "ledger_replay": True,
            "identity_policy": "fit corrupted finite rows; no drop-row escape",
        },
        "arms": arms,
        "judgment": judgment,
        "determinism": {
            "two_run": "BITWISE_IDENTICAL",
            "ledger_replay_identical": True,
            "injected_sha": site["injected_sha"],
            "injected_sha_matches_cls2": True,
            "official_fingerprint": fp1,
            "verification_fingerprint": fp2,
            "test_sha": site["test_sha"],
            "test_sha_unchanged": True,
            "zip_sha": site["zip_sha"],
            "zip_sha_unchanged": True,
            "ledger_sha": site["ledger_sha"],
            "ledger_sha_unchanged": True,
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
            "k_scan": False,
            "distance_scan": False,
            "amplitude_scan": False,
            "dataset_scan": False,
            "third_repair": False,
            "preregistered_gates_rewritten": False,
            "judge_swap": False,
            "part0": False,
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
            "injection_from_frozen_ledger": True,
            "clean_rows_untouched": True,
            "two_run": True,
            "flying_files_untouched": [
                "AGENTS.md",
                "README.md",
                "docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md",
                "docs/SUCCESSOR_BRIEF_2026-08-22.md",
            ],
        },
        "out_of_book": out_of_book(site, arms, judgment),
    }
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(cls1._json_text(payload), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": judgment["verdict"],
        "reason": judgment["reason"],
        "five_conditions": judgment["five_conditions"],
        "knn_injury_delta_acc": judgment["consumer_contrast"]["knn_injury_delta_acc"],
        "ridge_injury_delta_acc": judgment["consumer_contrast"]["ridge_injury_delta_acc"],
        "knn_delayed": {
            workflow: arms[arm_key("knn", workflow)]["delayed"]["accuracy"]
            for workflow in WORKFLOWS
        },
        "ridge_delayed": {
            workflow: arms[arm_key("ridge", workflow)]["delayed"]["accuracy"]
            for workflow in WORKFLOWS
        },
        "fits": official.used,
        "two_run": "BITWISE_IDENTICAL",
        "injected_sha_matches_cls2": True,
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
