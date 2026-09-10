"""M-R0c: the same thirty steps, read a second way -- through the repaired path.

Read-only.  0 external LLM, 0 new Consumer fits, 0 held-out reads, 0 sealed
reads; no production code, threshold, budget, candidate space or information
wall is touched, and no existing artifact is overwritten.

Run:  python -m evaluation.main_protocol_p4.audit_m_r0c_repaired_eligibility

Three additions to M-R0b, and nothing else
-------------------------------------------
1. **as-run beside repaired.**  M-R0b answered "was a Scope modification
   feasible" on the row set the live course searched.  This adds the repaired
   caliber: the evidence the current ``census`` actually supplies (exact
   deduplication, unit votes), the candidate the current
   ``propose_candidates`` actually emits given the Active set, Draft state,
   current/root Scope and the revision and verification counts **as they stood
   at that step**, and the shadow plus the existing structural narrowing check
   on that candidate.  Both are reported side by side with the difference
   named.  Deduplication can move a record either way, so all ninety-six are
   recomputed rather than only the seven M-R0b found.

2. **Does the *whole* modified Scope still match later?**  The full Scope is
   the candidate's own base predicate plus the shadow clause.  Coverage on
   later units is computed through the production resolver on deployment-
   visible context only.  "The old program was deployed later and treated five
   series" is *not* used as a proxy: a narrowed Scope resolves to a different
   set, and that set is what has to clear the coverage floor.

3. **One verification is not a chain.**  A later unit that could carry the
   Support and delayed faces is reported apart from the question of whether
   *another* unit exists after it for a re-encounter.  With one later unit
   there is no chain-timing opportunity, however eligible that unit is.

What stays NOT_EVALUATED
------------------------
Whether Slow would name the clause, whether the replay screen would pass it,
whether a later unit would verify it, and what any of it would gain.  Coverage
eligibility is a statement about which series the predicate resolves, not about
any gate.  Records the course did not keep are UNKNOWN, never a pass or a fail.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import audit_m_r0b_revision_opportunity as opp
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import scope_narrowing_preflight as narrowing
from evaluation.main_protocol_p4 import scope_threshold_tool as tool

ART = base.ART
PERIOD = base.PERIOD_K_UNITS
MATERIAL = base.MATERIAL
MIN_TREATED = base.MIN_TREATED
NOT_EVALUATED = opp.NOT_EVALUATED
UNKNOWN = opp.UNKNOWN

#: Support face and the delayed authority, from the runner.  The evaluation
#: face (+144) is deliberately absent: it may not take part in any selection.
DELAYED_OFFSET = 48
EVALUATION_OFFSET_EXCLUDED = 144


# --------------------------------------------------------------------------
# the step-time ledger, rebuilt so the repaired propose path can be run on it
# --------------------------------------------------------------------------

def _ledger_at(timelines: list[dict[str, Any]], position: int,
               key_for) -> tuple[drafts.DraftLedger, dict[str, Any]]:
    """A ledger holding this arm's Drafts as they stood at ``position``.

    Reconstructed, not simulated: every state, attempt count and closure comes
    from the recorded event the course wrote.  The census key is filled in
    because that is what the repaired ``restrict`` records; the historical
    Drafts carry ``None`` there, which is the very defect the repair closed.
    """
    ledger = drafts.DraftLedger()
    notes = []
    for timeline in timelines:
        state = opp._draft_state_at(timeline, position)
        if not state.get("exists"):
            notes.append({"draft_id": timeline["draft_id"],
                          "at_this_step": state.get("why")})
            continue
        draft = drafts.RestrictedDraft(
            draft_id=timeline["draft_id"],
            program_steps=tuple((step["op"], dict(step.get("params") or {}))
                                for step in _steps_of(timeline["program"])),
            root_scope=dict(timeline["root_scope"]),
            # No Draft in this course was ever revised, so the predicate it
            # carries at any step is still the one it was restricted with.
            current_scope=dict(timeline["root_scope"]),
            revisions=0,
            created_at_origin=0,
            census_key=key_for(timeline),
        )
        draft.state = state["state"]
        draft.verification_attempts = int(state["verification_attempts"])
        draft.history.append({"event": "verification",
                              "state_after": state["state"]})
        if state.get("closed"):
            draft.closed = state["closed"]
        ledger.drafts.append(draft)
        notes.append({"draft_id": draft.draft_id, "state": draft.state,
                      "verification_attempts": draft.verification_attempts,
                      "closed": draft.closed})
    return ledger, {"drafts_at_this_step": notes}


def _steps_of(program: str) -> list[dict[str, Any]]:
    return [{"op": op, "params": {}} for op in str(program).split(">")]


# --------------------------------------------------------------------------
# the shadow, and the legality check the outer loop already applies
# --------------------------------------------------------------------------

def _shadow_and_legality(rows: list[dict[str, Any]],
                         base_scope: Any, root_scope: Any) -> dict[str, Any]:
    existing = list((base_scope or {}).get("predicate") or ())
    found = opp._shadow(rows, {"predicate": existing})
    out = {"rows_searched": len(rows), **found}
    if found.get("feasible") is not True or not found.get("clause"):
        out["full_scope"] = None
        out["narrowing_preflight"] = None
        return out
    predicate = [dict(clause) for clause in existing]
    predicate.append(dict(found["clause"]))
    full = {"scope_type": "serving_series_predicate", "predicate": predicate}
    out["full_scope"] = full
    # The same call ``consolidate`` makes: the structural half, with the root
    # so the lifecycle clause budget is counted against the initialiser.
    out["narrowing_preflight"] = narrowing.validate_narrowing(
        dict(base_scope or {}), full, root=dict(root_scope or {}) or None
    ).to_dict()
    return out


# --------------------------------------------------------------------------
# coverage of the *whole* modified Scope on later units
# --------------------------------------------------------------------------

def _coverage(doc: dict[str, Any], arm: str, after: int,
              full_scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve the modified predicate on every later unit of this arm.

    Deployment-visible context only, through the runner's own resolver, at the
    Support origin and at the delayed origin.  The evaluation face is never
    resolved here: it may not take part in choosing or qualifying anything.
    """
    rows = []
    for cell in base.cells(doc, arm):
        position = int(cell["position"])
        if position <= after:
            continue
        unit = cell["unit"]
        ctx = opp._unit_ctx(unit)
        support = ctx.resolve(full_scope, int(unit["origin"]))
        delayed = ctx.resolve(full_scope, int(unit["origin"]) + DELAYED_OFFSET)
        rows.append({
            "position": position,
            "unit": "%s@%d" % (unit["block"], int(unit["origin"])),
            "origin": int(unit["origin"]),
            "delayed_origin": int(unit["origin"]) + DELAYED_OFFSET,
            "served": len(ctx.eval_uids),
            "treated_on_support": len(support),
            "treated_on_delayed": len(delayed),
            "clears_the_coverage_floor_on_delayed": len(delayed) >= MIN_TREATED,
            "support_side_floor_applied": False,
            "why_no_support_floor": (
                "the coverage floor is a line in the delayed authoritative "
                "gate; Support admission (admission_policy.decide) has no "
                "treated>=N condition at all.  treated_on_support is reported "
                "for information and is not a qualification: whether Support "
                "would admit is NOT_EVALUATED and needs a real reading"),
            "support_admission_outcome": NOT_EVALUATED,
            "evaluation_face_used": False,
            "verification_outcome": NOT_EVALUATED,
        })
    return rows


def _chain(coverage: list[dict[str, Any]]) -> dict[str, Any]:
    """One verification unit, and then whether a re-encounter unit follows it."""
    eligible = [row for row in coverage
                if row["clears_the_coverage_floor_on_delayed"]]
    first = eligible[0] if eligible else None
    after = [row for row in eligible
             if first and row["position"] > first["position"]]
    return {
        "later_units": len(coverage),
        "later_units_clearing_the_floor_under_the_full_scope": len(eligible),
        "has_a_verification_unit": bool(first),
        "verification_unit": (
            {"position": first["position"], "unit": first["unit"],
             "treated_on_delayed": first["treated_on_delayed"]}
            if first else None),
        "has_a_re_encounter_unit_after_it": bool(after),
        "re_encounter_units_after_it": [
            {"position": row["position"], "unit": row["unit"],
             "treated_on_delayed": row["treated_on_delayed"]}
            for row in after],
        "full_chain_timing_opportunity": bool(first and after),
        "why_not": (None if (first and after)
                    else ("no later unit resolves at or above the coverage "
                          "floor under the modified Scope" if not first
                          else "only one later unit qualifies, so a "
                               "verification leaves nothing to re-encounter")),
        "support_side_floor_applied": False,
        "coverage_is_not_a_pass": (
            "clearing the floor means the predicate resolves enough series to "
            "take a reading; whether the reading passes is NOT_EVALUATED"),
    }


# --------------------------------------------------------------------------

#: Runner constants, imported as numbers rather than restated as prose.
FITS_PER_SCORED_FACE = 3
CACHE_FITS_PER_CELL = 2
SUPPORT_FITS_PER_PROBE = 2
OUTER_LLM_PER_STEP = 2
FAST_LLM_PER_CELL_CAP = 5


def _cost(with_chain: list[dict[str, Any]]) -> dict[str, Any]:
    """What one real acceptance of these chains would cost, itemised.

    An estimate, not an authorisation, and not a budget request: every line
    says where its number comes from, and the parts that were never measured
    say so instead of carrying a plausible figure.  Cache savings are listed
    on their own and are **not** subtracted from any ceiling -- a saving is
    not a bound.
    """
    rows = []
    for row in with_chain:
        chain = row["chain_timing"]
        processed = row["boundary_position"] + 1
        later = chain["later_units_clearing_the_floor_under_the_full_scope"]
        rows.append({
            "lineage": "%s|%s|%s" % (row["ordering"], row["arm"],
                                     row["census_key"]),
            "k_index": row["k_index"],
            "outer_slow_physical_calls": {
                "per_step_cap": OUTER_LLM_PER_STEP,
                "one_attempt_costs": 1,
                "calls_for_one_complete_proposal": UNKNOWN,
                "why": ("no proposal in the recorded course ever reached "
                        "CALIBRATED, so the cost of a complete one has never "
                        "been measured (M-R0)"),
            },
            "replay_screen_fits": {
                "cells_this_arm_had_processed": processed,
                "fits_per_cell_through_the_cache": CACHE_FITS_PER_CELL,
                "estimate": processed * CACHE_FITS_PER_CELL,
                "basis": "run_hec1.CACHE_FITS_PER_CELL x processed cells",
            },
            "verification_unit_fits": {
                "units": 1,
                "delayed_face": FITS_PER_SCORED_FACE,
                "support_probe_rate": SUPPORT_FITS_PER_PROBE,
                "probes_that_unit_would_take": NOT_EVALUATED,
                "why": ("the probe count depends on what Fast proposes on "
                        "that unit, which is not decided here"),
            },
            "re_encounter_unit_fits": {
                "units_available": len(chain["re_encounter_units_after_it"]),
                "delayed_face_each": FITS_PER_SCORED_FACE,
                "support_probe_rate": SUPPORT_FITS_PER_PROBE,
                "probes_each": NOT_EVALUATED,
            },
            "ancestor_shadow_contrast_fits": {
                "later_units_clearing_the_floor": later,
                "extra_fits_per_face": 1,
                "faces_per_unit": 2,
                "estimate_if_every_one_is_scored": later * 2,
                "basis": ("the execution plan's ancestor shadow: v1 scored on "
                          "the same face and series as v2, one extra fit per "
                          "face, billed"),
            },
            "shared_baseline": {
                "evaluations_per_unit_touched": 1,
                "priced": False,
                "why": ("cached per (executor, origin) and read by every arm "
                        "on that unit; how a shared model should be charged "
                        "is open (R3), so it is counted and not priced"),
            },
            "fast_path_llm": {
                "per_cell_cap": FAST_LLM_PER_CELL_CAP,
                "cells_involved": 1 + len(chain["re_encounter_units_after_it"]),
                "calls_actually_needed": NOT_EVALUATED,
            },
        })
    return {
        "scope": ("one lineage carried from its outer step through one "
                  "verification unit and one re-encounter unit"),
        "is_not_an_authorisation": (
            "these are arithmetic from recorded constants and recorded unit "
            "counts; nothing here approves a run, and no ceiling is implied"),
        "cache_savings_listed_apart": {
            "replay_prediction_cache": (
                "the replay screen's per-cell cost is already the cached rate "
                "(2 rather than 3); no further saving is assumed"),
            "shared_baseline": (
                "one reference evaluation per unit serves every arm on it"),
            "not_deducted_from_any_ceiling": True,
        },
        "unknown_costs": [
            "the physical calls one complete structural proposal takes",
            "how many Support probes each involved unit would take",
            "whether any of it would pass, which changes nothing about the "
            "cost but everything about what the cost buys",
        ],
        "per_chain": rows,
    }


def build() -> dict[str, Any]:
    k0 = json.loads(base.K0_RECEIPT.read_text(encoding="utf-8"))
    k0_keys = list(k0.get("lineage_keys") or ())
    records: list[dict[str, Any]] = []

    for name in base.ORDERINGS:
        doc = base.load(name)
        for arm in base.ONLINE_ARMS:
            timelines = opp._draft_timeline(doc, arm)
            held: set[str] = set(k0_keys) if arm.startswith("A5") else set()
            bank: list[dict[str, Any]] = []
            snaps: dict[int, dict[str, Any]] = {}
            for cell in base.cells(doc, arm):
                position = int(cell["position"])
                if cell.get("lineage_key"):
                    held.add(str(cell["lineage_key"]))
                for row in base.bank_rows_for_cell(cell):
                    bank.append(opp._bank_row(cell, row))
                if (position + 1) % PERIOD == 0:
                    snaps[(position + 1) // PERIOD] = {
                        "boundary_position": position,
                        "held": set(held), "bank": list(bank)}

            for step in [row for row in doc["outer_steps"]
                         if row["arm"] == arm]:
                k = int(step["k_index"])
                snap = snaps.get(k)
                if snap is None:
                    continue
                groups = outer_loop.census(snap["bank"], material=MATERIAL)
                by_key = {str(group["census_key"]): group for group in groups}

                def key_for(timeline, _by=by_key):
                    signature = outer_loop._root_scope_signature(
                        timeline["root_scope"])
                    for candidate_key, group in _by.items():
                        if (group["program_signature"].split("(")[0]
                                == timeline["program"]
                                and group["root_scope_signature"] == signature):
                            return candidate_key
                    return None

                ledger, ledger_note = _ledger_at(
                    timelines, snap["boundary_position"], key_for)
                emitted, signals = outer_loop.propose_candidates(
                    groups, ledger=ledger,
                    held_lineage_keys=sorted(snap["held"]))
                emitted_by_program: dict[str, list[dict[str, Any]]] = {}
                for candidate in emitted:
                    emitted_by_program.setdefault(
                        str(candidate["program_signature"]).split("(")[0],
                        []).append(candidate)

                for group in groups:
                    program = str(group["program_signature"]).split("(")[0]
                    key = str(group["census_key"])
                    as_run_rows = opp._rows_as_run(snap["bank"], group)
                    as_run = opp._shadow(as_run_rows,
                                         group.get("root_scope"))
                    as_run["rows_searched"] = len(as_run_rows)

                    mine = emitted_by_program.get(program) or []
                    repaired: dict[str, Any] = {
                        "candidate_emitted": bool(mine),
                        "kinds": [row["kind"] for row in mine],
                        "refusals": [row.get("outcome_preset") for row in mine
                                     if row.get("outcome_preset")],
                        "why_refused": [row.get("why_refused") for row in mine
                                        if row.get("why_refused")],
                    }
                    clause_needing = [row for row in mine
                                      if row.get("needs_clause")
                                      and not row.get("outcome_preset")]
                    repaired["has_a_clause_needing_branch"] = bool(clause_needing)
                    repaired["has_a_refusal_branch"] = bool(repaired["refusals"])
                    repaired["carries_both_branches"] = bool(
                        clause_needing and repaired["refusals"])
                    if clause_needing:
                        candidate = clause_needing[0]
                        repaired["kind"] = candidate["kind"]
                        repaired["draft_id"] = candidate.get("draft_id")
                        repaired["base_scope"] = candidate.get(
                            "base_scope") or candidate.get("root_scope")
                        repaired["root_scope"] = candidate.get("root_scope")
                        repaired.update(_shadow_and_legality(
                            list(candidate.get("rows") or ()),
                            repaired["base_scope"], repaired["root_scope"]))
                    else:
                        repaired["full_scope"] = None
                        repaired["feasible"] = False
                        repaired["outcome"] = (
                            "NO_CLAUSE_NEEDING_CANDIDATE_EMITTED")
                        repaired["rows_searched"] = 0

                    difference = []
                    if as_run.get("rows_searched") != repaired.get(
                            "rows_searched"):
                        difference.append(
                            "rows searched %s -> %s (exact deduplication)"
                            % (as_run.get("rows_searched"),
                               repaired.get("rows_searched")))
                    if as_run.get("outcome") != repaired.get("outcome"):
                        difference.append("shadow outcome %s -> %s"
                                          % (as_run.get("outcome"),
                                             repaired.get("outcome")))
                    if (as_run.get("clause") or repaired.get("clause")) and (
                            as_run.get("clause") != repaired.get("clause")):
                        difference.append("clause %s -> %s"
                                          % (as_run.get("clause"),
                                             repaired.get("clause")))
                    if as_run.get("feasible_count") != repaired.get(
                            "feasible_count"):
                        difference.append("feasible stumps %s -> %s"
                                          % (as_run.get("feasible_count"),
                                             repaired.get("feasible_count")))

                    eligible = None
                    chain = None
                    preflight = repaired.get("narrowing_preflight") or {}
                    if repaired.get("full_scope") and preflight.get("accepted"):
                        eligible = _coverage(doc, arm,
                                             snap["boundary_position"],
                                             repaired["full_scope"])
                        chain = _chain(eligible)

                    records.append({
                        "ordering": name, "arm": arm, "k_index": k,
                        "boundary_position": snap["boundary_position"],
                        "census_key": key, "program": program,
                        "active_ancestor_at_this_step": key in snap["held"],
                        "adverse_units": group["adverse_units"],
                        "positive_units": group["positive_units"],
                        "unit_votes": group["unit_votes"],
                        "observations": group["observations"],
                        "bank_rows_seen": group["bank_rows_seen"],
                        "duplicate_observations_dropped": group[
                            "duplicate_observations_dropped"],
                        "ledger_at_this_step": ledger_note,
                        "as_run": as_run,
                        "repaired": repaired,
                        "difference": difference or ["none"],
                        "later_coverage_under_the_full_scope": eligible,
                        "chain_timing": chain,
                        "not_evaluated": {
                            "slow_would_name_this_clause": NOT_EVALUATED,
                            "replay_screen": NOT_EVALUATED,
                            "later_verification_outcome": NOT_EVALUATED,
                            "gain_on_re_encounter": NOT_EVALUATED,
                        },
                    })

    both_searched = [row for row in records
                     if row["as_run"].get("rows_searched", 0) > 0
                     and row["repaired"].get("rows_searched", 0) > 0]
    differed = [row for row in both_searched if row["difference"] != ["none"]]

    qualified = [row for row in records
                 if (row["repaired"].get("narrowing_preflight") or {}).get(
                     "accepted")]
    with_verification = [row for row in qualified
                         if row["chain_timing"]["has_a_verification_unit"]]
    with_chain = [row for row in qualified
                  if row["chain_timing"]["full_chain_timing_opportunity"]]

    blocked = Counter()
    for row in records:
        repaired = row["repaired"]
        if not repaired["candidate_emitted"]:
            blocked["no candidate emitted by the repaired propose path"] += 1
        elif repaired.get("has_a_clause_needing_branch"):
            # Classified by the branch that is still live.  One record carries
            # a refusal *and* a clause-needing branch, because two Drafts of
            # one lineage existed at that step; putting it in the refusal
            # bucket would hide where it actually stopped.
            if repaired.get("feasible") is not True:
                blocked["no feasible clause: %s" % repaired.get("outcome")] += 1
            elif not (repaired.get("narrowing_preflight") or {}).get("accepted"):
                blocked["narrowing preflight refused: %s"
                        % (repaired.get("narrowing_preflight") or {}).get(
                            "reason", "")[:48]] += 1
            elif not row["chain_timing"]["has_a_verification_unit"]:
                blocked["no later unit clears the floor under the full Scope"] += 1
            elif not row["chain_timing"]["full_chain_timing_opportunity"]:
                blocked["only one later unit qualifies: no chain timing"] += 1
        else:
            blocked["refused at propose: %s" % repaired["refusals"][0]] += 1

    def _keys(rows, fn):
        return sorted({fn(row) for row in rows})

    return {
        "stage": "M_R0C_REPAIRED_ELIGIBILITY",
        "written_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "evidence_grade": "INSTRUMENT -- an eligibility table, not a result",
        "boundary": {
            "external_llm_calls": 0, "new_consumer_fits": 0,
            "held_out_reads": 0, "sealed_reads": 0,
            "evaluation_face_reads_used_in_selection": 0,
            "production_code_changed": 0, "thresholds_changed": 0,
            "existing_artifacts_written": 0, "new_sha_or_hash_introduced": 0,
        },
        "version_discipline": base.version_check(),
        "builds_on": {
            "m_r0": "artifacts/main_protocol/m_r0_reachability.json",
            "m_r0b": "artifacts/main_protocol/m_r0b_revision_opportunity.json",
            "as_run_shadow_validated": (
                "M-R0b reproduced all three recorded live shadows; that "
                "validation is not repeated here and is not superseded"),
        },
        "totals": {
            "step_lineage_records": len(records),
            "repaired_candidate_emitted": sum(
                1 for row in records if row["repaired"]["candidate_emitted"]),
            "branch_census": {
                "with_a_clause_needing_branch": sum(
                    1 for row in records
                    if row["repaired"].get("has_a_clause_needing_branch")),
                "with_a_refusal_branch": sum(
                    1 for row in records
                    if row["repaired"].get("has_a_refusal_branch")),
                "carrying_both_branches": sum(
                    1 for row in records
                    if row["repaired"].get("carries_both_branches")),
                "identity": ("emitted = clause-needing + refusal - both; the "
                             "two are not mutually exclusive, so neither may "
                             "be reported as 'the rest of' the other"),
                "where_the_overlap_is": [
                    "%s|%s|k%d|%s" % (row["ordering"], row["arm"],
                                      row["k_index"], row["program"])
                    for row in records
                    if row["repaired"].get("carries_both_branches")],
                "why_it_overlaps": (
                    "two Drafts of one lineage existed at that step -- one "
                    "closed, one still REVISABLE -- so the narrowing was "
                    "refused against the closed shell while the open one was "
                    "proposed as a REVISE"),
            },
            "repaired_clause_feasible": sum(
                1 for row in records
                if row["repaired"].get("feasible") is True),
            "repaired_clause_legal": len(qualified),
            "with_a_later_verification_unit": len(with_verification),
            "with_a_full_chain_timing_opportunity": len(with_chain),
            "distinct_lineages_with_a_verification_unit": _keys(
                with_verification,
                lambda row: "%s|%s|%s" % (row["ordering"], row["arm"],
                                          row["census_key"])),
            "distinct_lineages_with_a_full_chain": _keys(
                with_chain,
                lambda row: "%s|%s|%s" % (row["ordering"], row["arm"],
                                          row["census_key"])),
            "distinct_identities_with_a_verification_unit": _keys(
                with_verification, lambda row: row["census_key"]),
            "distinct_identities_with_a_full_chain": _keys(
                with_chain, lambda row: row["census_key"]),
            "records_where_both_calibers_actually_searched": len(both_searched),
            "records_whose_shadow_differed_when_both_searched": len(differed),
            "shadow_difference_note": (
                "a record where the repaired path emitted no clause-needing "
                "candidate has nothing to search, so it is not counted as a "
                "shadow difference; the 96 - %d others differ only in that"
                % len(both_searched)),
            "blocking_distribution": dict(
                sorted(blocked.items(), key=lambda row: -row[1])),
        },
        "minimum_acceptance_cost_estimate": _cost(with_chain),
        "records": records,
    }


def main() -> int:
    report = build()
    out = ART / "m_r0c_repaired_eligibility.json"
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8")).get("stage")
        except (OSError, ValueError):
            existing = None
        if existing != report["stage"]:
            sys.stderr.write("refusing to overwrite %s\n" % out)
            return 2
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print("wrote %s" % out)
    for key, value in report["totals"].items():
        if isinstance(value, list):
            print("  %-52s %d" % (key, len(value)))
        elif isinstance(value, dict):
            print("  %s:" % key)
            for name, count in value.items():
                print("      %-50s %s" % (name[:50], count))
        else:
            print("  %-52s %s" % (key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
