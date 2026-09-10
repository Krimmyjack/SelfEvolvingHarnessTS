"""Step 1 of the input/role contrast: what did Slow actually see, and does the
deterministic search still find a clause in *that* slice?

Zero LLM calls, zero Consumer fits.  The k=1 outer step is rebuilt exactly the
way the live runner rebuilds it (``run_m_r0d_forward_k1_outer_step._state_at_k1``
is imported, not re-implemented), ``consolidate`` is driven with a Slow that
records the candidate it is handed and then abstains -- which is the same
``payload is None`` path the live attempt-3 call took, so the outer-step record
this produces is the same one, minus the transport.

The asymmetry this exists to measure is in ``run_hec1.OuterSlowAgent.__call__``:

    "evidence_rows": [... for row in rows[:60]]

while ``outer_loop._clause_for`` hands ``tool.clause_from_slow`` -- and through
it the ScopeFit shadow -- ``candidate["rows"]`` **whole**.  So the shadow and
the model were not reading the same evidence, and the earlier note that "Slow
saw the same 100 rows" was wrong.  The truncation length is read out of the
source here rather than restated, so this report cannot drift from the code.

What this can and cannot settle:

* It **can** say whether a deterministic search restricted to the model's own
  slice still finds a feasible clause, and whether the damage the task asks
  about is even present in that slice.
* It **cannot** say the truncation caused the abstention.  One live call, one
  model, one step; the model was not asked again on the full rows.  That is
  step 2, and it is not run here.

Run:  python -m evaluation.main_protocol_p4.audit_m_r0e_slow_input_truncation
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import scope_threshold_tool as tool

ART = base.ART
POLICY = tool.BOUNDED_RISK_V1


# ---------------------------------------------------------------------------
# the truncation, read out of the production source
# ---------------------------------------------------------------------------

def _truncation() -> dict[str, Any]:
    """``rows[:N]`` as it is actually written in ``OuterSlowAgent.__call__``."""
    source = inspect.getsource(runner.OuterSlowAgent.__call__)
    found = re.search(r"for\s+row\s+in\s+rows\[:(\d+)\]", source)
    if not found:
        return {"present": False, "limit": None,
                "why": "no rows[:N] slice found in OuterSlowAgent.__call__"}
    lines = inspect.getsourcelines(runner.OuterSlowAgent.__call__)
    offset = lines[1]
    hit = next(i for i, text in enumerate(lines[0]) if found.group(0) in text)
    return {
        "present": True,
        "limit": int(found.group(1)),
        "literal": found.group(0),
        "site": "%s:%d" % ("evaluation/main_protocol_p4/run_hec1.py",
                           offset + hit),
        "field": "public_input['evidence_rows']",
        "the_tool_is_not_truncated": (
            "outer_loop._clause_for passes candidate['rows'] whole to "
            "tool.clause_from_slow, and the ScopeFit shadow inside it searches "
            "that whole set"),
    }


# ---------------------------------------------------------------------------
# a Slow that costs nothing
# ---------------------------------------------------------------------------

class _CaptureSlow:
    """Records the candidate it is asked about, then takes the abstain path.

    Returning ``None`` is byte-for-byte the branch the live attempt-3 call
    reached (``payload is None`` -> ``SLOW_ABSTAINED``), so the step record
    below is the same record, produced without a transport.
    """

    def __init__(self) -> None:
        self.captured: list[Mapping[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, candidate: Mapping[str, Any],
                 rejected: Sequence[Mapping[str, Any]]) -> None:
        self.captured.append(candidate)
        self.calls.append({"candidate": candidate.get("kind"),
                           "outcome": "CAPTURED_THEN_ABSTAINED",
                           "no_llm": True})
        return None


def _refusing_replay(**_kwargs: Any) -> dict[str, Any]:
    raise AssertionError("the replay screen must not be reached: 0 fits")


_refusing_replay.estimated_fits_per_candidate = 0


# ---------------------------------------------------------------------------
# the two searches
# ---------------------------------------------------------------------------

def _composition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    units: list[str] = []
    for row in rows:
        unit = str(row.get("unit"))
        if unit not in units:
            units.append(unit)
    per_unit = {unit: sum(1 for row in rows if str(row.get("unit")) == unit)
                for unit in units}
    gains = [float(row.get("gain") or 0.0) for row in rows]
    harmed = [row for row in rows
              if float(row.get("gain") or 0.0) < -POLICY.material]
    harmed_units: list[str] = []
    for row in harmed:
        unit = str(row.get("unit"))
        if unit not in harmed_units:
            harmed_units.append(unit)
    return {
        "rows": len(rows),
        "units": units,
        "unit_count": len(units),
        "rows_per_unit": per_unit,
        "series": sorted({str(row.get("series")) for row in rows}),
        "aggregate_gain": (round(sum(gains) / len(gains), 6)
                           if gains else None),
        "harmed_rows": len(harmed),
        "harmed_fraction": (round(len(harmed) / len(gains), 4)
                            if gains else None),
        "units_with_harm": harmed_units,
        "worst_single_row_harm": (round(max(0.0, -min(gains)), 6)
                                  if gains else None),
    }


def _feasible_set(rows: Sequence[Mapping[str, Any]],
                  existing: Sequence[Mapping[str, Any]],
                  vocabulary: Sequence[str]) -> dict[str, Any]:
    """Every (feature, direction, edge) the tool would call feasible here.

    Same enumeration ``best_stump`` performs, kept separately so the report can
    show *which* line each rejected triple failed on rather than only a count.
    """
    surviving = tool._existing_survivors(rows, existing)
    feasible: list[dict[str, Any]] = []
    line_failures: dict[str, int] = {}
    considered = 0
    for feature in vocabulary:
        try:
            edges = tool._edges_for(str(feature), None)
        except Exception:  # noqa: BLE001 - not in the numeric vocabulary
            continue
        for direction in tool.DIRECTIONS:
            for edge in edges:
                reading = tool._reading(
                    tool._select(surviving, str(feature), direction, edge),
                    POLICY)
                considered += 1
                if reading["feasible"]:
                    feasible.append({"feature": str(feature),
                                     "direction": direction,
                                     "threshold": float(edge),
                                     "treated": reading["treated"],
                                     "aggregate_gain":
                                         reading["aggregate_gain"]})
                    continue
                for line, ok in reading["lines"].items():
                    if not ok:
                        line_failures[line] = line_failures.get(line, 0) + 1
    return {
        "rows_the_scope_already_selects": len(surviving),
        "triples_considered": considered,
        "feasible_count": len(feasible),
        "feasible": sorted(feasible,
                           key=lambda row: (-(row["aggregate_gain"] or 0.0),
                                            row["feature"],
                                            row["direction"],
                                            row["threshold"])),
        "distinct_feasible_features": sorted(
            {row["feature"] for row in feasible}),
        "line_that_failed_counts": dict(sorted(line_failures.items())),
    }


def _search(rows: Sequence[Mapping[str, Any]],
            existing: Sequence[Mapping[str, Any]],
            vocabulary: Sequence[str], label: str) -> dict[str, Any]:
    stump = tool.best_stump(rows=rows, policy=POLICY,
                            vocabulary=vocabulary,
                            existing_clauses=existing)
    return {
        "input": label,
        "composition": _composition(rows),
        "best_stump": stump,
        "feasible_set": _feasible_set(rows, existing, vocabulary),
    }


def _transfer(winner: Mapping[str, Any],
              rows: Sequence[Mapping[str, Any]],
              existing: Sequence[Mapping[str, Any]],
              vocabulary: Sequence[str]) -> dict[str, Any]:
    """Would the full-input winner have calibrated on the model's slice?"""
    if winner.get("outcome") != "BEST_STUMP":
        return {"applicable": False,
                "why": "the full input produced no best stump"}
    try:
        got = tool.calibrate(feature=str(winner["feature"]),
                             direction=str(winner["direction"]),
                             rows=rows, policy=POLICY,
                             vocabulary=vocabulary,
                             existing_clauses=existing)
    except tool.NoFeasibleThreshold as exc:
        return {"applicable": True, **exc.to_dict()}
    except Exception as exc:  # noqa: BLE001
        return {"applicable": True, "outcome": "ERROR",
                "error": "%s: %s" % (type(exc).__name__, exc)}
    return {"applicable": True,
            "outcome": got["outcome"],
            "feature": got["feature"],
            "direction": got["direction"],
            "threshold": got["threshold"],
            "treated": got["treated"],
            "same_threshold_as_full_input": (
                float(got["threshold"]) == float(winner["threshold"])),
            "reading": got["reading"]}


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def build() -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    doc = base.load(live.ORDERING)
    state = live._state_at_k1(doc)

    slow = _CaptureSlow()
    # Identical to the live attempt so the path is the same one; the screen is
    # never reached because Slow abstains, and _refusing_replay proves it.
    budget = outer_loop.OuterBudget(
        outer_llm_per_step=int(live.contract.OUTER_LLM_PER_STEP),
        replay_fits_remaining=live.MAX_CONSUMER_FITS)
    record = outer_loop.consolidate(
        bank=state["bank"], ledger=state["ledger"], k_index=live.K_INDEX,
        slow=slow, replay=_refusing_replay, budget=budget,
        held_lineage_keys=state["held"])
    step = record.to_dict()

    truncation = _truncation()
    limit = truncation.get("limit")
    vocabulary = list(live.contract.SCOPE_CLASS["vocabulary"])

    per_candidate: list[dict[str, Any]] = []
    for candidate in slow.captured:
        rows = [dict(row) for row in (candidate.get("rows") or ())]
        existing = list(dict(candidate.get("base_scope") or {}).get("predicate")
                        or ())
        seen = rows[:limit] if limit else rows
        dropped = rows[limit:] if limit else []
        full = _search(rows, existing, vocabulary, "full candidate['rows']")
        slice_ = _search(seen, existing, vocabulary,
                         "the model's slice rows[:%s]" % limit)
        tail = (_search(dropped, existing, vocabulary,
                        "the dropped tail rows[%s:]" % limit)
                if dropped else None)
        per_candidate.append({
            "kind": candidate.get("kind"),
            "draft_id": candidate.get("draft_id"),
            "program_signature": candidate.get("program_signature"),
            "base_scope": dict(candidate.get("base_scope") or {}),
            "existing_clauses": existing,
            "why": candidate.get("why"),
            "rows_total": len(rows),
            "rows_shown_to_the_model": len(seen),
            "rows_withheld_from_the_model": len(dropped),
            "the_model_saw": slice_,
            "the_tool_searched": full,
            "the_dropped_tail_alone": tail,
            "full_winner_on_the_models_slice": _transfer(
                full["best_stump"], seen, existing, vocabulary),
            "verdict": _verdict(full, slice_),
        })

    return {
        "stage": "M_R0E_SLOW_INPUT_TRUNCATION",
        "question": (
            "step 1 of the contrast: run the same deterministic search on the "
            "rows the model actually received and compare it with the full set"
        ),
        "written_at": started.isoformat(),
        "cost": {"llm_calls": 0, "consumer_fits": 0,
                 "how": ("the Slow stub records the candidate and returns None, "
                         "which is the same abstain branch the live call took; "
                         "the replay screen raises if reached and was not")},
        "corrects": {
            "artifact": "m_r0d_forward_k1_outer_step_live__attempt3.md",
            "section": "五、一个值得记录的对照",
            "was": "Slow 在看到同样的 100 行后回答 insufficient_public_evidence",
            "is": ("the model received rows[:%s]; the shadow searched all %s. "
                   "The contrast was not on equal inputs."
                   % (limit, per_candidate[0]["rows_total"]
                      if per_candidate else "?")),
            "found_by": "non-author review",
        },
        "truncation": truncation,
        "policy": POLICY.to_dict(),
        "vocabulary": vocabulary,
        "target": {"ordering": live.ORDERING, "arm": live.ARM,
                   "k_index": live.K_INDEX,
                   "boundary_position": live.BOUNDARY_POSITION,
                   "bank_rows": len(state["bank"]),
                   "units_in_bank": len(state["processed"])},
        "candidates_put_to_slow": len(slow.captured),
        "per_candidate": per_candidate,
        "outer_step_record": step,
        "matches_the_live_attempt": _cross_check(step, per_candidate),
        "not_settled_here": [
            "whether the truncation caused the abstention -- the model was not "
            "asked again on the full rows; that is step 2",
            "whether the Slow role instruction (the general preparation "
            "Agent's 'do not infer candidate utility') contributed",
            "whether any clause found here has real downstream benefit -- no "
            "later unit was touched",
        ],
        "code_state": base.version_check(),
    }


def _verdict(full: Mapping[str, Any], seen: Mapping[str, Any]) -> dict[str, Any]:
    full_ok = full["best_stump"].get("outcome") == "BEST_STUMP"
    seen_ok = seen["best_stump"].get("outcome") == "BEST_STUMP"
    if full_ok and not seen_ok:
        name = "SIGNAL_LOST_IN_THE_TRUNCATION"
        why = ("a deterministic search finds a feasible clause on the full "
               "rows and none on the slice the model received")
    elif full_ok and seen_ok:
        same = (full["best_stump"].get("feature"),
                full["best_stump"].get("direction")) == (
                    seen["best_stump"].get("feature"),
                    seen["best_stump"].get("direction"))
        name = ("SIGNAL_PRESENT_IN_BOTH__SAME_WINNER" if same
                else "SIGNAL_PRESENT_IN_BOTH__DIFFERENT_WINNER")
        why = ("the slice the model received also supports a feasible clause; "
               "truncation alone does not explain the abstention")
    elif not full_ok and not seen_ok:
        name = "NO_FEASIBLE_CLAUSE_EITHER_WAY"
        why = "the deterministic search finds nothing on either input"
    else:
        name = "FEASIBLE_ONLY_ON_THE_SLICE"
        why = "unexpected: the slice supports a clause the full set does not"
    return {"verdict": name, "why": why,
            "this_is_not_a_causal_claim": (
                "the model was not re-asked on the full rows in this audit")}


def _cross_check(step: Mapping[str, Any],
                 per_candidate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Confirm this 0-cost replay reproduces the live attempt-3 step."""
    path = ART / "m_r0d_forward_k1_outer_step_live__attempt3.json"
    if not path.exists():
        return {"checked": False, "why": "no attempt-3 receipt on disk"}
    live_doc = json.loads(path.read_text(encoding="utf-8"))
    live_step = live_doc.get("outer_step") or {}
    live_cands = list(live_step.get("candidates") or ())
    here = list(step.get("candidates") or ())
    return {
        "checked": True,
        "candidate_count": {"live": len(live_cands), "here": len(here)},
        "same_candidate_count": len(live_cands) == len(here),
        "why_strings_match": [
            (a.get("why") == b.get("why"))
            for a, b in zip(live_cands, here)],
        "outcomes": {"live": [c.get("outcome") for c in live_cands],
                     "here": [c.get("outcome") for c in here]},
        "bank_rows_in_the_revise_evidence": {
            "live": [(c.get("evidence") or {}).get("bank_rows")
                     for c in live_cands],
            "here": [row["rows_total"] for row in per_candidate],
        },
    }


def main(argv: list[str] | None = None) -> int:
    report = build()
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "m_r0e_slow_input_truncation.json"
    if out.exists():
        print("refusing to overwrite %s" % out)
        return 2
    text = json.dumps(drafts._plain(report), indent=1, ensure_ascii=False,
                      default=str)
    out.write_text(text, encoding="utf-8")
    print("wrote %s" % out)
    for row in report["per_candidate"]:
        print(json.dumps({
            "kind": row["kind"],
            "rows_total": row["rows_total"],
            "shown": row["rows_shown_to_the_model"],
            "withheld": row["rows_withheld_from_the_model"],
            "full_best": row["the_tool_searched"]["best_stump"].get("clause")
            or row["the_tool_searched"]["best_stump"].get("outcome"),
            "full_feasible": row["the_tool_searched"]["feasible_set"][
                "feasible_count"],
            "slice_best": row["the_model_saw"]["best_stump"].get("clause")
            or row["the_model_saw"]["best_stump"].get("outcome"),
            "slice_feasible": row["the_model_saw"]["feasible_set"][
                "feasible_count"],
            "verdict": row["verdict"]["verdict"],
        }, ensure_ascii=False, indent=1))
    print(json.dumps(report["matches_the_live_attempt"], ensure_ascii=False,
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
