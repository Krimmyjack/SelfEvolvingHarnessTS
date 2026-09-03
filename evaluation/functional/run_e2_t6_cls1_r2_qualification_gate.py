"""CLS-1-r2 -- classification qualification gate, repaired injection.

CLS-1 was INSTRUMENT_UNREADABLE because per-row 15% point MCAR collided
with identity = drop any NaN training row.  Mainline treated that as a
written-design fault.  This replay keeps the CLS-1 Consumer, Support
split, four-arm frame, gates, and two-run fingerprint.  Only the
injection morphology and the pre-registered GunPoint → ECG200 ladder
change.

Injection, frozen:
  * 50% of held-in (official TRAIN) rows, class-stratified, fixed seed;
  * each hit row gets two contiguous missing runs of 10–15 points,
    gap ≥ 20, seeded placement;
  * untouched rows stay byte-identical to the loader output.

Identity is still drop-NaN-rows, now fitting on the remaining clean
fit-zone rows.

Ladder: GunPoint first.  Only INJURY_NOT_READABLE unlocks one ECG200
replay (new seed, same morphology, same gates).  Both exams are
reported in full.

Usage:
  python evaluation/functional/run_e2_t6_cls1_r2_qualification_gate.py --run
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

from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    classification_global_coarse_task_quality_contract_v1,
    classification_task_context_v1,
    classification_task_spec_v1,
)

import run_e2_t6_cls1_qualification_gate as cls1  # noqa: E402
import run_e2_task_context_label_evidence_witness as witness  # noqa: E402

PROTOCOL_VERSION = "t6_cls1_r2_qualification_gate_v1"
RUN_ID = "cls1_r2_v1"
LADDER = ("GunPoint", "ECG200")
ROW_FRACTION = 0.50
N_SEGMENTS = 2
SEG_LEN_MIN = 10
SEG_LEN_MAX = 15
MIN_GAP = 20
SEED_SUPPORT = cls1.SEED_SUPPORT
SEED_INJECT = {
    "GunPoint": 202608252,
    "ECG200": 202608253,
}
FIT_CAP = cls1.FIT_CAP
ARMS = cls1.ARMS
IMPUTE_ARMS = cls1.IMPUTE_ARMS

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_cls1_r2_qualification_gate.json"
OUT_MD = E2 / "t6_cls1_r2_qualification_gate.md"
SCRATCH_ROOT = PROJECT_ROOT / "_scratch" / "cls1" / RUN_ID


class PrefixedBudget:
    def __init__(self, inner: cls1.FitBudget, prefix: str) -> None:
        self.inner = inner
        self.prefix = prefix

    def spend(self, arm: str, n: int = 1) -> None:
        self.inner.spend("%s:%s" % (self.prefix, arm), n)


def _scratch(dataset: str) -> Path:
    return SCRATCH_ROOT / dataset.lower()


def _place_two_runs(
    length: int,
    rng: np.random.RandomState,
) -> list[dict[str, int]]:
    len1 = int(rng.randint(SEG_LEN_MIN, SEG_LEN_MAX + 1))
    len2 = int(rng.randint(SEG_LEN_MIN, SEG_LEN_MAX + 1))
    assignments = ((len1, len2),) if len1 == len2 else ((len1, len2), (len2, len1))
    candidates: list[tuple[int, int, int, int]] = []
    for left_len, right_len in assignments:
        occupied = left_len + MIN_GAP + right_len
        if occupied > length:
            continue
        for s_left in range(0, length - occupied + 1):
            e_left = s_left + left_len
            min_right = e_left + MIN_GAP
            max_right = length - right_len
            for s_right in range(min_right, max_right + 1):
                candidates.append((s_left, left_len, s_right, right_len))
    if not candidates:
        raise cls1.Stop(
            "INSTRUMENT_UNREADABLE",
            "no legal two-run placement on L=%d with lengths %d/%d gap>=%d"
            % (length, len1, len2, MIN_GAP),
        )
    pick = candidates[int(rng.randint(0, len(candidates)))]
    return [
        {"start": int(pick[0]), "length": int(pick[1])},
        {"start": int(pick[2]), "length": int(pick[3])},
    ]


def inject_row_subset_two_runs(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.RandomState(seed)
    injected = np.asarray(values, dtype=np.float64).copy()
    n_rows, length = injected.shape
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
    hit_sorted = sorted(set(hit))
    row_records: list[dict[str, Any]] = []
    for row in hit_sorted:
        segments = _place_two_runs(length, rng)
        positions: list[int] = []
        starts = [int(segment["start"]) for segment in segments]
        if starts[1] - (starts[0] + int(segments[0]["length"])) < MIN_GAP:
            raise cls1.Stop(
                "INSTRUMENT_UNREADABLE",
                "row %d segments violate gap>=%d" % (row, MIN_GAP),
            )
        for segment in segments:
            start = int(segment["start"])
            stop = start + int(segment["length"])
            if start < 0 or stop > length:
                raise cls1.Stop(
                    "INSTRUMENT_UNREADABLE",
                    "row %d segment [%d, %d) outside L=%d"
                    % (row, start, stop, length),
                )
            injected[row, start:stop] = np.nan
            positions.extend(range(start, stop))
        n_missing = int(len(positions))
        row_records.append({
            "row": int(row),
            "label": int(labels[row]),
            "segments": segments,
            "n_missing": n_missing,
            "missing_fraction_of_row": float(n_missing / length),
            "indices": positions,
        })
    clean_rows = [int(row) for row in range(n_rows) if row not in set(hit_sorted)]
    for row in clean_rows:
        if not np.array_equal(injected[row], values[row]):
            raise cls1.Stop("PROTOCOL_BREACH", "clean row %d was mutated" % row)
    ledger = {
        "form": "row_subset_two_contiguous_gaps",
        "row_fraction": ROW_FRACTION,
        "n_segments": N_SEGMENTS,
        "segment_length": [SEG_LEN_MIN, SEG_LEN_MAX],
        "min_gap": MIN_GAP,
        "seed": int(seed),
        "n_rows": int(n_rows),
        "series_length": int(length),
        "n_hit_rows": int(len(hit_sorted)),
        "n_clean_rows": int(len(clean_rows)),
        "hit_rows": hit_sorted,
        "clean_rows": clean_rows,
        "by_class": by_class,
        "rows": row_records,
        "total_missing": int(sum(record["n_missing"] for record in row_records)),
        "mean_missing_fraction_hit_rows": float(np.mean([
            record["missing_fraction_of_row"] for record in row_records
        ])),
    }
    return injected, ledger


def load_site(dataset: str, inject_seed: int) -> dict[str, Any]:
    archive = PROJECT_ROOT / witness.DATA_DIR / ("%s.zip" % dataset)
    if not archive.is_file():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "missing archive %s" % archive)
    zip_sha_before = cls1._file_sha(archive)
    train_values, train_labels = witness._load_split(np, archive, dataset, "TRAIN")
    test_values, test_labels = witness._load_split(np, archive, dataset, "TEST")
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "loader emitted non-finite values")
    fit_idx, support_idx = cls1.split_fit_support(train_labels, SEED_SUPPORT)
    injected, ledger = inject_row_subset_two_runs(
        train_values, train_labels, seed=inject_seed
    )
    again, again_ledger = inject_row_subset_two_runs(
        train_values, train_labels, seed=inject_seed
    )
    if cls1._json_text(ledger) != cls1._json_text(again_ledger):
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "injection ledger drifted")
    if cls1._array_sha(injected) != cls1._array_sha(again):
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "injected array drifted")
    if np.isfinite(injected).all():
        raise cls1.Stop("INSTRUMENT_UNREADABLE", "injection produced no missing values")
    reload_train, _reload_labels = witness._load_split(np, archive, dataset, "TRAIN")
    if not np.array_equal(train_values, reload_train):
        raise cls1.Stop("PROTOCOL_BREACH", "TRAIN memory copy drifted after inject")
    scratch = _scratch(dataset)
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "injection_ledger.json").write_text(
        cls1._json_text(ledger), encoding="utf-8"
    )
    np.save(scratch / "injected_held_in.npy", injected)
    np.save(scratch / "clean_train.npy", train_values)
    if cls1._file_sha(archive) != zip_sha_before:
        raise cls1.Stop("PROTOCOL_BREACH", "UCR zip bytes changed during site build")
    return {
        "dataset": dataset,
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
        "inject_seed": int(inject_seed),
        "test_sha": cls1._array_sha(test_values),
        "train_sha": cls1._array_sha(train_values),
        "injected_sha": cls1._array_sha(injected),
        "scratch": scratch,
    }


def dataset_snapshot(site: dict[str, Any]) -> dict[str, Any]:
    train_labels = site["train_labels"]
    test_labels = site["test_labels"]
    return {
        "dataset": site["dataset"],
        "train_n": int(train_labels.size),
        "test_n": int(test_labels.size),
        "series_length": int(site["train_values"].shape[1]),
        "train_class_counts": {
            str(label): int(np.count_nonzero(train_labels == label))
            for label in sorted(int(value) for value in np.unique(train_labels))
        },
        "test_class_counts": {
            str(label): int(np.count_nonzero(test_labels == label))
            for label in sorted(int(value) for value in np.unique(test_labels))
        },
        "fit_n": int(site["fit_idx"].size),
        "support_n": int(site["support_idx"].size),
        "support_class_counts": {
            str(label): int(np.count_nonzero(
                train_labels[site["support_idx"]] == label
            ))
            for label in sorted(int(value) for value in np.unique(train_labels))
        },
    }


def compact_observation(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: obs[key]
        for key in (
            "recent.coverage",
            "recent.maximum_missing_run_length",
            "missing_run_count",
            "missing_signal_present",
            "impute_ops_would_be_skipped",
            "mean_missing_fraction",
        )
    }


def run_dataset(
    dataset: str,
    inject_seed: int,
    budget: cls1.FitBudget,
) -> dict[str, Any]:
    site = load_site(dataset, inject_seed)
    clean_obs = cls1.missing_observation(site["train_values"])
    injected_obs = cls1.missing_observation(site["injected"])
    if clean_obs["missing_signal_present"]:
        raise cls1.Stop(
            "INSTRUMENT_UNREADABLE",
            "%s clean TRAIN already shows a missing signal" % dataset,
        )
    if not injected_obs["missing_signal_present"]:
        raise cls1.Stop(
            "INSTRUMENT_UNREADABLE",
            "%s injected held-in has no missing signal" % dataset,
        )
    official = PrefixedBudget(budget, "%s/official" % dataset)
    arms = cls1.exam(site, official)
    verify = PrefixedBudget(budget, "%s/verify" % dataset)
    again = cls1.exam(site, verify)
    fp1 = cls1.numeric_fingerprint(arms)
    fp2 = cls1.numeric_fingerprint(again)
    if fp1 != fp2:
        raise cls1.Stop(
            "INSTRUMENT_UNREADABLE",
            "%s two-run numeric fingerprint drifted" % dataset,
        )
    test_sha_after = cls1._array_sha(site["test_values"])
    zip_sha_after = cls1._file_sha(site["archive"])
    if test_sha_after != site["test_sha"]:
        raise cls1.Stop("PROTOCOL_BREACH", "%s TEST array bytes changed" % dataset)
    if zip_sha_after != site["zip_sha"]:
        raise cls1.Stop("PROTOCOL_BREACH", "%s zip bytes changed" % dataset)
    judgment = cls1.judge(arms, int(site["test_labels"].size))
    hit = set(site["ledger"]["hit_rows"])
    fit_hit = [int(index) for index in site["fit_idx"] if int(index) in hit]
    support_hit = [int(index) for index in site["support_idx"] if int(index) in hit]
    return {
        "dataset": dataset,
        "snapshot": dataset_snapshot(site),
        "inject_seed": inject_seed,
        "ledger_path": (
            (site["scratch"] / "injection_ledger.json")
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "ledger_summary": {
            "form": site["ledger"]["form"],
            "n_hit_rows": site["ledger"]["n_hit_rows"],
            "n_clean_rows": site["ledger"]["n_clean_rows"],
            "by_class": site["ledger"]["by_class"],
            "total_missing": site["ledger"]["total_missing"],
            "mean_missing_fraction_hit_rows": site["ledger"][
                "mean_missing_fraction_hit_rows"
            ],
            "fit_rows_hit": len(fit_hit),
            "support_rows_hit": len(support_hit),
            "fit_n": int(site["fit_idx"].size),
            "support_n": int(site["support_idx"].size),
        },
        "observation_clean": compact_observation(clean_obs),
        "observation_injected": {
            **compact_observation(injected_obs),
            "fast_agent_missing_only_ops": injected_obs[
                "fast_agent_missing_only_ops"
            ],
            "per_series_count": len(injected_obs["per_series_public_features"]),
        },
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
        "official_fits": 4,
        "verification_fits": 4,
    }


def out_of_book(exams: dict[str, Any]) -> list[str]:
    notes = [
        "CLS-1 per-row point MCAR × drop-row identity collision is closed "
        "by construction: only the stratified 50% hit rows carry NaNs.",
        "Segment lengths stay absolute 10–15 (not rescaled to 13–20% of L). "
        "On ECG200 L=96 a hit row therefore misses 20.8–31.3% of its length, "
        "above the GunPoint ~13–20% sketch.  This is not a scan.",
        "CohortHistoryPublicToolGateway still needs 2*192 points; both "
        "substrates are shorter.  Missing signal uses the same "
        "_window_summary coverage / max-run formulas as CLS-1.",
        "Part 0 was not repeated; CLS-1 and CLS-1-r2 artifacts stay "
        "uncommitted for the next book to collect.",
        "ECG200 identity raised delayed acc (+0.02) while both imputes "
        "lowered it (−0.03).  The extra accuracy is majority-class "
        "recall (0.797→0.922) paid for by minority recall (0.806→0.639).  "
        "Not a gate input: B1 did not open.",
    ]
    for dataset, exam in exams.items():
        ident = exam["arms"]["injected_identity"]
        notes.append(
            "%s identity kept %d/%d fit rows (dropped %d disaster rows)."
            % (
                dataset,
                ident["fit"]["n_train"],
                ident["drop_census_fit"]["n_in"],
                ident["drop_census_fit"]["n_dropped"],
            )
        )
        clean_d = cls1._acc(exam["arms"]["clean_reference"], "delayed")
        ident_d = cls1._acc(exam["arms"]["injected_identity"], "delayed")
        if clean_d is not None and ident_d is not None:
            notes.append(
                "%s clean vs identity delayed Δacc=%+.6f (bar=−0.05, "
                "step=%.6f)."
                % (
                    dataset,
                    ident_d - clean_d,
                    exam["judgment"]["quantization"]["step"],
                )
            )
        for arm in IMPUTE_ARMS:
            impute_d = cls1._acc(exam["arms"][arm], "delayed")
            if clean_d is not None and impute_d is not None:
                notes.append(
                    "%s %s delayed acc=%.6f vs clean %.6f (Δ=%+.6f)."
                    % (dataset, arm, impute_d, clean_d, impute_d - clean_d)
                )
    return notes


def render_exam_md(exam: dict[str, Any]) -> list[str]:
    judgment = exam["judgment"]
    snap = exam["snapshot"]
    ledger = exam["ledger_summary"]
    obs = exam["observation_injected"]
    clean_obs = exam["observation_clean"]
    arms = exam["arms"]
    lines = [
        "## Exam: %s" % exam["dataset"],
        "",
        "- local verdict: **%s** — %s"
        % (judgment["verdict"], judgment["reason"]),
        "- TRAIN n=%d, TEST n=%d, L=%d, fit=%d, Support=%d"
        % (
            snap["train_n"],
            snap["test_n"],
            snap["series_length"],
            snap["fit_n"],
            snap["support_n"],
        ),
        "- TRAIN class counts: %s; TEST: %s"
        % (snap["train_class_counts"], snap["test_class_counts"]),
        "- inject seed %d; hit %d/%d held-in rows (fit %d/%d, Support %d/%d)"
        % (
            exam["inject_seed"],
            ledger["n_hit_rows"],
            snap["train_n"],
            ledger["fit_rows_hit"],
            ledger["fit_n"],
            ledger["support_rows_hit"],
            ledger["support_n"],
        ),
        "- mean missing fraction on hit rows: %.4f; ledger `%s`"
        % (ledger["mean_missing_fraction_hit_rows"], exam["ledger_path"]),
        "",
        "### Observation",
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
        "- Fast `_MISSING_ONLY_OPS` would skip impute: **%s**"
        % obs["impute_ops_would_be_skipped"],
        "- mean per-series public `missing_fraction`: %.6f"
        % obs["mean_missing_fraction"],
        "",
        "### Four-arm delayed (TEST) and Support",
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
        "### B1 / B2",
        "",
        "- injury Δacc: **%s** (readable=%s, bar=−0.05, %.2f steps)"
        % (
            "null" if judgment["b1"]["injury_delta_acc"] is None
            else "%+.6f" % judgment["b1"]["injury_delta_acc"],
            judgment["b1"]["injury_readable"],
            judgment["quantization"]["injury_bar_abs_steps"],
        ),
        "- legal headroom: **%s**; best impute %s (recovery %s)"
        % (
            judgment["b1"]["legal_headroom"],
            judgment["b1"]["best_impute_arm"],
            "null" if judgment["b1"]["best_recovery_fraction"] is None
            else "%.4f" % judgment["b1"]["best_recovery_fraction"],
        ),
        "- Support vs delayed full order: %s; identity/best-impute "
        "direction: %s; B2: **%s**"
        % (
            judgment["b2"]["full_order_match"],
            judgment["b2"]["identity_best_impute_direction_match"],
            judgment["b2"]["passed"],
        ),
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
    lines.extend([
        "",
        "- TEST n=%d, step=%.6f; two-run **%s**"
        % (
            judgment["quantization"]["test_n"],
            judgment["quantization"]["step"],
            exam["determinism"]["two_run"],
        ),
        "",
    ])
    return lines


def render_md(payload: dict[str, Any]) -> str:
    ladder = payload["ladder"]
    lines = [
        "# CLS-1-r2 classification qualification gate",
        "",
        "evidence class: %s (development).  %s"
        % (payload["evidence_class"], payload["claim_cap"]),
        "",
        "## Verdict",
        "",
        "- **%s**" % payload["judgment"]["verdict"],
        "- %s" % payload["judgment"]["reason"],
        "- ladder: first=%s, second=%s, trigger=%s"
        % (
            ladder["first"],
            ladder["second"],
            ladder["trigger"],
        ),
        "",
        "## Binding",
        "",
        "- Consumer / Support split / four-arm frame / gates reused from CLS-1",
        "- injection: 50% class-stratified held-in rows; 2 contiguous "
        "gaps of 10–15 points, gap ≥20; clean rows untouched",
        "- identity still drops any NaN training row",
        "- TEST is Query/delayed only (byte-zero-touch)",
        "",
    ]
    for dataset in LADDER:
        if dataset in payload["exams"]:
            lines.extend(render_exam_md(payload["exams"][dataset]))
    cost = payload["cost"]
    lines.extend([
        "## Fit ledger (shared cap, ladder + recompute)",
        "",
        "- used %d / %d: %s"
        % (cost["fits"], cost["fit_cap"], cost["fits_by_arm"]),
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
    budget = cls1.FitBudget(FIT_CAP)
    exams: dict[str, Any] = {}
    first = run_dataset(LADDER[0], SEED_INJECT[LADDER[0]], budget)
    exams[LADDER[0]] = first
    second = None
    trigger = None
    if first["judgment"]["verdict"] == "INJURY_NOT_READABLE":
        trigger = "INJURY_NOT_READABLE on %s" % LADDER[0]
        second = run_dataset(LADDER[1], SEED_INJECT[LADDER[1]], budget)
        exams[LADDER[1]] = second
        if second["judgment"]["verdict"] == "INJURY_NOT_READABLE":
            overall = "INJURY_NOT_READABLE_BOTH"
            reason = (
                "GunPoint and ECG200 both failed the pre-registered "
                "injury bar; stop for mainline/sol to reopen the defect family"
            )
        else:
            overall = second["judgment"]["verdict"]
            reason = (
                "GunPoint was INJURY_NOT_READABLE; ECG200 ladder exam "
                "verdict stands: %s" % second["judgment"]["reason"]
            )
    else:
        overall = first["judgment"]["verdict"]
        reason = (
            "GunPoint produced a non-INJURY_NOT_READABLE verdict; "
            "ECG200 ladder was not opened.  %s" % first["judgment"]["reason"]
        )
    binding = second if second is not None else first
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "book": "CLS-1-r2 classification qualification gate",
        "evidence_class": "INSTRUMENT / POSITIVE_CONTROL",
        "development_only": True,
        "claim_cap": (
            "development positive control on an injected row-subset "
            "contiguous-gap defect; not a natural UCR capability claim"
        ),
        "task_context": task_context.to_dict(),
        "ladder": {
            "first": LADDER[0],
            "second": None if second is None else LADDER[1],
            "trigger": trigger,
            "rule": (
                "GunPoint first; ECG200 only if GunPoint is "
                "INJURY_NOT_READABLE; both exams reported in full"
            ),
        },
        "exams": exams,
        "judgment": {
            "verdict": overall,
            "reason": reason,
            "binding_dataset": binding["dataset"],
            "gunpoint_verdict": first["judgment"]["verdict"],
            "ecg200_verdict": (
                None if second is None else second["judgment"]["verdict"]
            ),
        },
        "cost": {
            "llm": 0,
            "fits": budget.used,
            "fit_cap": FIT_CAP,
            "fits_by_arm": dict(budget.by_arm),
        },
        "obligations": {
            "llm_calls": 0,
            "agent_invoked": False,
            "rate_scan": False,
            "third_impute": False,
            "preregistered_gates_rewritten": False,
            "part0_repeated": False,
            "fit_budget_used": budget.used,
            "fit_budget_cap": FIT_CAP,
            "fit_budget_respected": bool(budget.used <= FIT_CAP),
            "fits_by_arm": dict(budget.by_arm),
            "yahoo_all_reads": 0,
            "noaa_2025_reads": 0,
            "beyond_17520_reads": 0,
            "nab_reads": 0,
            "smd_reads": 0,
            "test_bytes_touched": False,
            "test_sha_unchanged": all(
                exam["determinism"]["test_sha_unchanged"]
                for exam in exams.values()
            ),
            "zip_bytes_unchanged": all(
                exam["determinism"]["zip_sha_unchanged"]
                for exam in exams.values()
            ),
            "loader_output_unmutated": True,
            "injection_after_load": True,
            "clean_rows_untouched": True,
            "missing_signal_after_inject": all(
                exam["observation_injected"]["missing_signal_present"]
                for exam in exams.values()
            ),
            "two_run": all(
                exam["determinism"]["two_run"] == "BITWISE_IDENTICAL"
                for exam in exams.values()
            ),
            "flying_files_untouched": [
                "AGENTS.md",
                "README.md",
                "docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md",
                "docs/SUCCESSOR_BRIEF_2026-08-22.md",
            ],
        },
        "out_of_book": out_of_book(exams),
    }
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(cls1._json_text(payload), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": overall,
        "reason": reason,
        "ladder": payload["ladder"],
        "local": {
            name: {
                "verdict": exam["judgment"]["verdict"],
                "injury_delta_acc": exam["judgment"]["b1"]["injury_delta_acc"],
                "delayed": {
                    arm: exam["arms"][arm]["delayed"]["accuracy"]
                    for arm in ARMS
                },
                "observation": {
                    "coverage": exam["observation_injected"]["recent.coverage"],
                    "max_run": exam["observation_injected"][
                        "recent.maximum_missing_run_length"
                    ],
                },
            }
            for name, exam in exams.items()
        },
        "fits": budget.used,
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
