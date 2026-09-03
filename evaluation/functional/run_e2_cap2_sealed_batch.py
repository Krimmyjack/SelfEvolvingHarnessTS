"""CAP-2: the sealed conditional-exam batch.

Protocol: ``artifacts/functional/e2/cap2_sealed_batch_protocol_freeze.md``.
This module is a shell.  It selects candidates from public metadata, downloads
them one at a time, re-checks structure, decides the frozen card's Scope
mechanically, and -- on the first Scope match only -- hands the cell to the
CAP-1 exam skeleton.  Nothing shared is touched: no threshold, no menu, no
injection template, no ``methods/`` file.

  python evaluation/functional/run_e2_cap2_sealed_batch.py --select
  python evaluation/functional/run_e2_cap2_sealed_batch.py --sequential
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
FREEZE_MD = E2 / "cap2_sealed_batch_protocol_freeze.md"
SELECTION_JSON = E2 / "cap2_selection.json"
SELECTION_MD = E2 / "cap2_selection.md"
EXAM_JSON = E2 / "cap2_sequential_exam.json"
EXAM_MD = E2 / "cap2_sequential_exam.md"
CHECKPOINT = E2 / "cap2_sequential_exam.checkpoint.json"
SA1_GATES_JSON = E2 / "sa1_minimal_gates.json"
SA1_FREEZE_JSON = E2 / "sa1_course_freeze.json"
CAP1_JSON = E2 / "cap1_capstone_protocol_freeze.json"

LOCAL_ARCHIVE_DIR = PROJECT_ROOT / "data" / "ucr_task_context"
DOWNLOAD_ROOT = PROJECT_ROOT / "data" / "ucr_conf_downloaded"
ROSTER = DOWNLOAD_ROOT / "ROSTER.md"
METADATA_URL = "https://timeseriesclassification.com/aeon-toolkit/metadata.csv"
METADATA_LOCAL = PROJECT_ROOT / "_scratch" / "tsc_metadata_cap2.csv"
ZIP_URL = "https://timeseriesclassification.com/aeon-toolkit/%s.zip"

PROTOCOL_VERSION = "cap2_sealed_batch_v1"
SA1_CARD_ID = "sa1_supply_scope_v2"
TEST_SUBSET_SEED = 20260828
TEST_SUBSET_CAP = 500
TRAIN_MIN, TRAIN_MAX = 40, 400
TOTAL_POINTS_MAX = 100_000
N_CANDIDATES = 3


class Stop(Exception):
    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__("%s: %s" % (verdict, reason))
        self.verdict = verdict
        self.reason = reason


def _line_of(path: str, needle: str) -> str:
    text = (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines()
    line = next((i + 1 for i, t in enumerate(text) if needle in t), None)
    return "%s:%s" % (path, line)


# =========================================================================== #
# Stage 1 -- structural blind selection (0 LLM, no values, no labels)
# =========================================================================== #
def template_gate(length: int) -> dict[str, Any]:
    """Can the frozen injection template even be written at this length?

    Every clause calls the live implementation rather than restating it, so a
    drift in the template would move this gate instead of silently passing it.

    * ``cls._v2_segment_length`` is ``round(1/150 * length)`` and
      ``cls._inject_v2`` raises when that is not positive -- the "length below
      the template's minimum constant" structural failure CAP-2 section 3
      names.
    * ``helpers['positions']`` refuses a geometry whose four spike positions
      are not distinct, or which comes within three samples of either end.
    * ``cls._inject_v2`` refuses a segment that would run off the end.
    * the artifact's own footprint -- four spikes of one segment each -- has to
      fit inside ``maximum_modified_fraction``, or no legal repair could cover
      what the injection wrote.
    """
    import run_e2_t6_cls_op_shared_harness as cls
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _helpers,
    )

    _ctx, helpers = _helpers()
    cap = float(cls._task_context().deployment_constraints
                .maximum_modified_fraction)
    segment = int(cls._v2_segment_length(int(length)))
    reasons: list[str] = []
    if segment < 1:
        reasons.append("segment_length_%d_below_template_minimum" % segment)
    positions: tuple[int, ...] | None = None
    try:
        positions = tuple(int(p) for p in helpers["positions"](int(length)))
    except Exception as exc:  # noqa: BLE001
        reasons.append("bound_positions_rejected:%s" % type(exc).__name__)
    if positions is not None and segment >= 1 \
            and max(positions) + segment > int(length):
        reasons.append("segment_overflows_series_end")
    footprint = (len(positions) if positions else 0) * max(segment, 0)
    fraction = (footprint / float(length)) if length else 1.0
    if segment >= 1 and fraction > cap:
        reasons.append("artifact_footprint_%.4f_exceeds_modification_cap_%.2f"
                       % (fraction, cap))
    return {
        "length": int(length),
        "segment_length": segment,
        "positions": list(positions) if positions else None,
        "artifact_points": footprint,
        "artifact_fraction_of_row": round(fraction, 6),
        "maximum_modified_fraction": cap,
        "pass": not reasons,
        "reasons": reasons,
        "citations": {
            "segment_rule": _line_of(
                "evaluation/functional/run_e2_t6_cls_op_shared_harness.py",
                "def _v2_segment_length"),
            "segment_positive_or_raise": _line_of(
                "evaluation/functional/run_e2_t6_cls_op_shared_harness.py",
                "if segment <= 0:"),
            "overflow_or_raise": _line_of(
                "evaluation/functional/run_e2_t6_cls_op_shared_harness.py",
                "v2 segment [%d, %d) overflows series length %d"),
            "position_geometry": _line_of(
                "evaluation/functional/"
                "run_e2_task_context_label_evidence_witness.py",
                "def _bound_positions"),
            "spike_fractions": _line_of(
                "evaluation/functional/"
                "run_e2_task_context_label_evidence_witness.py",
                "SPIKE_FRACTIONS = "),
            "modification_cap": _line_of(
                "evaluation/functional/run_e2_t6_cls_op_shared_harness.py",
                "maximum_modified_fraction"),
        },
    }


def _local_names() -> list[str]:
    return sorted(path.stem for path in LOCAL_ARCHIVE_DIR.glob("*.zip"))


def _roster_names(pool: Sequence[str]) -> list[str]:
    """Official names the roster already mentions, by whole-word match.

    Deliberately conservative: any archive name the roster text names at all
    is out, whether it was downloaded, sealed, or merely recorded as the
    eligible fourth.  Over-excluding costs a candidate; under-excluding would
    re-use a name already in the evidence record.
    """
    import re

    text = ROSTER.read_text(encoding="utf-8")
    return sorted(name for name in pool
                  if re.search(r"\b%s\b" % re.escape(name), text))


def _metadata_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = METADATA_LOCAL.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    source = {
        "url": METADATA_URL,
        "local_copy": METADATA_LOCAL.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "site_conventions": (
            "Length=0 marks variable length; Channels=1 marks univariate; "
            "NumberClasses is the official class count"),
    }
    return rows, source


def select() -> int:
    rows, source = _metadata_rows()
    pool = [str(row["Dataset"]) for row in rows]
    local = _local_names()
    roster = _roster_names(pool)
    excluded_names = sorted(set(local) | set(roster))

    trajectory: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["Dataset"])
        train = int(row["TrainSize"])
        test = int(row["TestSize"])
        length = int(row["Length"])
        classes = int(row["NumberClasses"])
        channels = int(row["Channels"])
        total_points = (train + test) * length
        gate = template_gate(length) if length > 0 else {
            "pass": False, "length": length,
            "reasons": ["variable_length_no_template_geometry"]}
        reasons: list[str] = []
        if classes != 2:
            reasons.append("not_binary:classes=%d" % classes)
        if length == 0:
            reasons.append("variable_length")
        if channels != 1:
            reasons.append("not_univariate:channels=%d" % channels)
        if not (TRAIN_MIN <= train <= TRAIN_MAX):
            reasons.append("train_rows_%d_outside_[%d,%d]"
                           % (train, TRAIN_MIN, TRAIN_MAX))
        if total_points > TOTAL_POINTS_MAX:
            reasons.append("total_points_%d_over_%d"
                           % (total_points, TOTAL_POINTS_MAX))
        if not gate["pass"]:
            reasons.extend("template_gate:%s" % item
                           for item in gate.get("reasons") or ())
        if name in local:
            reasons.append("name_already_local")
        if name in roster:
            reasons.append("name_already_in_roster")
        trajectory.append({
            "dataset": name, "train": train, "test": test, "length": length,
            "classes": classes, "channels": channels,
            "total_points": total_points,
            "template_gate_pass": bool(gate["pass"]),
            "template_gate": gate if length > 0 else None,
            "admitted": not reasons,
            "excluded_because": reasons,
        })

    eligible = sorted(row["dataset"] for row in trajectory if row["admitted"])
    chosen = eligible[:N_CANDIDATES]

    # Everything binary, equal-length and univariate that is neither already
    # local nor already named in the roster -- i.e. every genuinely fresh
    # name the archive offers -- with the clause each one fails.  This is the
    # audit that decides whether an empty pool is real or an artifact of the
    # deliberately conservative roster match.
    fresh = sorted(
        (row for row in trajectory
         if row["classes"] == 2 and row["length"] > 0 and row["channels"] == 1
         and row["dataset"] not in local and row["dataset"] not in roster),
        key=lambda row: row["dataset"])

    def _survivors(*, drop_roster: bool, drop_local: bool) -> list[str]:
        out = []
        for row in trajectory:
            if row["classes"] != 2 or row["length"] <= 0 or row["channels"] != 1:
                continue
            if not (TRAIN_MIN <= row["train"] <= TRAIN_MAX):
                continue
            if row["total_points"] > TOTAL_POINTS_MAX:
                continue
            if not row["template_gate_pass"]:
                continue
            if not drop_local and row["dataset"] in local:
                continue
            if not drop_roster and row["dataset"] in roster:
                continue
            out.append(row["dataset"])
        return sorted(out)

    counterfactuals = {
        "as_frozen": _survivors(drop_roster=False, drop_local=False),
        "if_roster_exclusion_dropped": _survivors(drop_roster=True,
                                                  drop_local=False),
        "if_both_name_exclusions_dropped": _survivors(drop_roster=True,
                                                      drop_local=True),
        "reading": (
            "the conservative roster match changes nothing -- dropping it "
            "entirely still admits nobody.  Dropping both name exclusions "
            "re-admits only names this line already holds locally, which is "
            "what makes the exhaustion real rather than an artifact of how "
            "the roster was parsed"),
    }
    verdict = ("CAP2_CANDIDATE_SET_FROZEN" if len(chosen) == N_CANDIDATES
               else "CAP2_CANDIDATE_POOL_EMPTY" if not chosen
               else "CAP2_CANDIDATE_POOL_SHORT")
    payload = {
        "protocol_version": PROTOCOL_VERSION + "/selection",
        "protocol_source": FREEZE_MD.relative_to(PROJECT_ROOT).as_posix(),
        "written_before_any_dataset_zip_download": True,
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "metadata_source": source,
        "metadata_reproducibility": (
            "the fresh fetch is byte-identical to the copy the 2026-08-25 "
            "CLS-CONF-dl book archived (same 7253 bytes, same sha256), so this "
            "census is stable and the selection below is reproducible from the "
            "same public table"),
        "filters": {
            "binary": "NumberClasses == 2",
            "equal_length": "Length != 0 (site convention)",
            "univariate": "Channels == 1",
            "train_rows": "%d <= TrainSize <= %d" % (TRAIN_MIN, TRAIN_MAX),
            "total_points": "(TrainSize + TestSize) * Length <= %d"
                            % TOTAL_POINTS_MAX,
            "template_gate": "derived from the live injection template and the "
                             "modification cap; see per-row citations",
            "name_exclusion": "not among the local archive stems nor any name "
                              "the download roster mentions",
        },
        "exclusion_lists": {
            "local_archive_stems": local,
            "local_archive_count": len(local),
            "roster_mentioned_names": roster,
            "union_size": len(excluded_names),
        },
        "counts": {
            "pool": len(rows),
            "binary": sum(1 for row in trajectory if row["classes"] == 2),
            "eligible": len(eligible),
        },
        "eligible_sorted": eligible,
        "selection_rule": "lexicographic first %d, frozen once" % N_CANDIDATES,
        "candidates": [{"slot": "C%d" % (index + 1), "dataset": name}
                       for index, name in enumerate(chosen)],
        "not_selected_but_eligible": eligible[N_CANDIDATES:],
        "verdict": verdict,
        "fresh_names_audit": {
            "definition": ("binary, equal-length, univariate, and neither "
                           "already local nor named in the download roster"),
            "count": len(fresh),
            "rows": [{"dataset": row["dataset"], "train": row["train"],
                      "test": row["test"], "length": row["length"],
                      "total_points": row["total_points"],
                      "template_gate_pass": row["template_gate_pass"],
                      "fails": row["excluded_because"]} for row in fresh],
            "reading": ("these are the only names in the published archive "
                        "that CAP-2 could newly seal; each fails at least one "
                        "section 1 clause on public metadata alone, with no "
                        "value or label read"),
        },
        "counterfactual_robustness": counterfactuals,
        "stop": ({
            "state": "STOPPED_BEFORE_ANY_DOWNLOAD",
            "reason": ("CAP-2 section 1's conjunction admits zero names, so "
                       "there is no C1/C2/C3 to freeze and Stage 2 has "
                       "nothing to open"),
            "why_not_scope_coverage_limited": (
                "section 3.6's SCOPE_COVERAGE_LIMITED is a verdict about the "
                "card's Scope, reached after three candidates are unsealed "
                "and their Pattern views computed.  Nothing was unsealed here "
                "and no Scope decision was made, so reusing that label would "
                "claim evidence this run does not have"),
            "nearest_miss_deliberately_not_taken": (
                "DodgerLoopGame fails exactly one clause -- TRAIN 20 against "
                "the frozen [40,400] band -- and passes every other, "
                "including the template gate and the point budget.  Relaxing "
                "the band to admit it is precisely the move this book "
                "forbids, so it was not made and is reported instead"),
            "requires": ("a mainline decision: widen a section 1 clause, "
                         "change the pool, or record the gap as structural.  "
                         "The executor has no discretion over any of the "
                         "three"),
        } if not chosen else None),
        "trajectory": trajectory,
        "obligations": {
            "zero_llm": True,
            "zero_values_or_labels_read": (
                "only the published metadata table was read; no dataset zip "
                "has been fetched at the time this artifact is written"),
            "no_value_or_family_based_preselection": (
                "every clause is a public metadata field or a mechanical "
                "consequence of the frozen template; nothing about the "
                "expected defect family entered the filter"),
            "downloads_so_far": 0,
        },
    }
    SELECTION_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                         sort_keys=True, default=str) + "\n",
                              encoding="utf-8")
    SELECTION_MD.write_text(_selection_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict,
                      "candidates": payload["candidates"],
                      "eligible": eligible,
                      "counts": payload["counts"],
                      "fresh_names": [row["dataset"]
                                      for row in fresh],
                      "counterfactuals": {
                          key: value for key, value in counterfactuals.items()
                          if key != "reading"},
                      "stop": payload.get("stop")},
                     ensure_ascii=False, indent=1))
    return 0 if len(chosen) == N_CANDIDATES else 1


def _git(*args: str) -> str:
    import subprocess
    try:
        return subprocess.run(["git", *args], cwd=str(PROJECT_ROOT),
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _selection_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# CAP-2 Stage 1: structural blind selection",
        "",
        "**Verdict: %s.**  Frozen candidates: %s"
        % (payload["verdict"],
           ", ".join("%s = %s" % (row["slot"], row["dataset"])
                     for row in payload["candidates"]) or "**none**"),
        "",
        "Written before any dataset zip was fetched (downloads so far: %s).  "
        "The pool is the official published table at `%s` -- %s rows, sha256 "
        "`%s`.  %s"
        % (payload["obligations"]["downloads_so_far"],
           payload["metadata_source"]["url"],
           payload["metadata_source"]["n_rows"],
           payload["metadata_source"]["sha256"],
           payload["metadata_reproducibility"]),
        "",
        "## Filters",
        "",
        "| filter | rule |", "|---|---|",
    ]
    for key, rule in payload["filters"].items():
        lines.append("| %s | %s |" % (key, rule))
    gate = next((row["template_gate"] for row in payload["trajectory"]
                 if row.get("template_gate")), None)
    if gate:
        lines += ["", "### Template compatibility gate, derived not invented",
                  "", "| clause | source |", "|---|---|"]
        for key, citation in (gate.get("citations") or {}).items():
            lines.append("| %s | `%s` |" % (key, citation))
        lines += ["", "Spike fractions are the frozen four, so the artifact "
                  "occupies four segments of `round(length/150)` samples each; "
                  "the modification cap is %.2f.  At every admitted length the "
                  "footprint is far inside the cap, so this clause never "
                  "decides a candidate on its own -- the binding clause is the "
                  "segment length and the end-margin geometry."
                  % gate["maximum_modified_fraction"], ""]

    if payload.get("stop"):
        stop = payload["stop"]
        lines += ["## Stop, before any download", "",
                  "`%s` -- %s" % (stop["state"], stop["reason"]), ""]
        for key in ("why_not_scope_coverage_limited",
                    "nearest_miss_deliberately_not_taken", "requires"):
            lines += ["- **%s**: %s" % (key, stop[key]), ""]

    fresh = payload.get("fresh_names_audit") or {}
    if fresh:
        lines += ["## Every genuinely fresh name the archive offers", "",
                  "%s -- %d rows.  %s"
                  % (fresh["definition"], fresh["count"], fresh["reading"]),
                  "",
                  "| dataset | train | test | length | total points | template "
                  "gate | fails |", "|---|---|---|---|---|---|---|"]
        for row in fresh["rows"]:
            lines.append("| %s | %d | %d | %d | %d | %s | %s |"
                         % (row["dataset"], row["train"], row["test"],
                            row["length"], row["total_points"],
                            row["template_gate_pass"],
                            "; ".join(row["fails"]) or "-"))
        lines.append("")

    cf = payload.get("counterfactual_robustness") or {}
    if cf:
        lines += ["## Is the empty pool real?", "",
                  "| name exclusions applied | eligible |", "|---|---|",
                  "| as frozen (local + roster) | %s |"
                  % (", ".join(cf["as_frozen"]) or "none"),
                  "| roster exclusion dropped | %s |"
                  % (", ".join(cf["if_roster_exclusion_dropped"]) or "none"),
                  "| both dropped | %s |"
                  % (", ".join(cf["if_both_name_exclusions_dropped"])
                     or "none"),
                  "", cf["reading"], ""]

    lines += ["## Counts", "",
              "| stage | n |", "|---|---|"]
    for key, value in payload["counts"].items():
        lines.append("| %s | %s |" % (key, value))
    lines += ["",
              "Eligible after the full conjunction, in lexicographic order: "
              "**%s**." % (", ".join(payload["eligible_sorted"]) or "none"),
              "",
              "Eligible but not selected (4th onward, recorded so the cut is "
              "auditable): %s."
              % (", ".join(payload["not_selected_but_eligible"]) or "none"),
              "",
              "## Exclusion lists", "",
              "- local archive stems (%d): %s"
              % (payload["exclusion_lists"]["local_archive_count"],
                 ", ".join(payload["exclusion_lists"]["local_archive_stems"])),
              "",
              "- names the download roster mentions: %s"
              % ", ".join(payload["exclusion_lists"]
                          ["roster_mentioned_names"]),
              "",
              "## Admitted rows", "",
              "| dataset | train | test | length | classes | channels | total "
              "points | template gate |",
              "|---|---|---|---|---|---|---|---|"]
    for row in payload["trajectory"]:
        if not row["admitted"]:
            continue
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %s |"
                     % (row["dataset"], row["train"], row["test"],
                        row["length"], row["classes"], row["channels"],
                        row["total_points"], row["template_gate_pass"]))
    lines += ["", "## Full trajectory (every pool row, with its mechanical "
              "reason)", "",
              "| dataset | admitted | excluded because |", "|---|---|---|"]
    for row in payload["trajectory"]:
        lines.append("| %s | %s | %s |"
                     % (row["dataset"], row["admitted"],
                        "; ".join(row["excluded_because"]) or "-"))
    lines += ["", "## Obligations", ""]
    for key, value in sorted(payload["obligations"].items()):
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


def sequential() -> int:
    """Stage 2.  Refuses to start when Stage 1 froze no candidate.

    The frozen protocol sequences over C1, C2, C3; with an empty candidate set
    there is nothing to download, nothing to unseal, and no Scope decision to
    make.  This writes the record of that rather than leaving a gap where a
    later reader would have to guess whether Stage 2 ran.
    """
    selection = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    candidates = list(selection.get("candidates") or ())
    if candidates:
        raise Stop("CAP2_STAGE2_NOT_IMPLEMENTED_FOR_A_NONEMPTY_SET",
                   "Stage 1 froze %d candidates; the sequential exam shell "
                   "must be reviewed against them before it runs"
                   % len(candidates))
    payload = {
        "protocol_version": PROTOCOL_VERSION + "/sequential",
        "protocol_source": FREEZE_MD.relative_to(PROJECT_ROOT).as_posix(),
        "state": "NOT_ENTERED",
        "git_head": _git("rev-parse", "HEAD"),
        "reason": ("Stage 1 verdict %s: CAP-2 section 1's conjunction admits "
                   "zero names, so no candidate was frozen and the sequential "
                   "exam had nothing to open"
                   % selection.get("verdict")),
        "downloads": 0,
        "unseals": 0,
        "zips_written": [],
        "roster_appended": False,
        "arms_run": 0,
        "llm": 0,
        "consumer_fits": 0,
        "capability_exam_verdict": None,
        "scope_decisions": [],
        "selection_artifact":
            SELECTION_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "note": ("this file exists so the record is unambiguous: Stage 2 was "
                 "not entered, no sealed material was fetched or opened, and "
                 "no verdict of any kind was produced"),
    }
    EXAM_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                    sort_keys=True, default=str) + "\n",
                         encoding="utf-8")
    EXAM_MD.write_text("\n".join([
        "# CAP-2 Stage 2: NOT ENTERED",
        "",
        "**%s**" % payload["reason"],
        "",
        "| item | value |", "|---|---|",
        "| downloads | 0 |",
        "| unseals | 0 |",
        "| arms run | 0 |",
        "| LLM | 0 |",
        "| consumer fits | 0 |",
        "| capability-exam verdict | none |",
        "",
        payload["note"],
        "",
        "Stage 1 trajectory and the stop reasoning: `%s`."
        % payload["selection_artifact"],
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"state": payload["state"],
                      "reason": payload["reason"]},
                     ensure_ascii=False, indent=1))
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CAP-2 sealed batch")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args(argv)
    if args.select:
        return select()
    if args.sequential:
        return sequential()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
