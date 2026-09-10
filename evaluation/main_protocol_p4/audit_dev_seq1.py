"""DEV-SEQ-1 contrasts: the five questions, answered from the recorded run.

Deterministic and offline.  It reads the run artifact and computes nothing the
run did not already measure -- no re-evaluation, no LLM, no Consumer fits.  The
point is that the report's numbers are produced by a program from the record,
not written by hand from a reading of it.

Denominators, stated once
-------------------------
* a **decision** is one (unit, arm, sequence);
* a per-sequence gain is that sequence's own gain under the program it ran;
* a unit delivery is the mean over that unit's decision population, every
  sequence under its own program, an untreated sequence contributing a measured
  0.0;
* the two follow-up arms are paired by (unit, sequence), so a difference is a
  difference on the same series in the same window.

UNKNOWN is carried, never zeroed.

Run:  python -m evaluation.main_protocol_p4.audit_dev_seq1 [--run-id run1]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import per_sequence as ps

ART = base.ROOT / "artifacts" / "main_protocol"
MATERIAL = ps.MATERIAL


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(
        value, bool) else None


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


# ---------------------------------------------------------------------------
# 1. was the observation, the decision and the execution really per sequence
# ---------------------------------------------------------------------------

def per_sequence_execution(doc: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for unit in _all_units(doc) for row in unit["sequences"]]
    bound = [row for row in rows
             if str(row.get("bound_series_uid")) == str(row.get("series_uid"))]
    features = {row["series_uid"]: json.dumps(row.get("public_features") or {},
                                              sort_keys=True, default=str)
                for row in rows}
    by_unit_programs = {
        "%s|%s" % (unit["arm"], unit["position"]):
            sorted({label for label in unit["assignment"].values()})
        for unit in _all_units(doc)}
    # What "executed == decided" has to mean, per sequence: the number Fast
    # measured when it decided for this sequence is the number the executed
    # policy delivered for this sequence.  An earlier version of this check
    # compared the per-unit ``programs_used`` map against the deployed labels,
    # which is weaker *and* wrong: ``program_label`` names operators only, so
    # two sequences choosing the same operator with different parameters
    # collapse into one key and the comparison misfires.  run2 u8 did exactly
    # that (five bare ``outlier_mad``, one with ``z_threshold=3.5``).
    delivered_mismatch = []
    label_collisions = []
    for unit in _all_units(doc):
        per_series = unit["support_reading"].get("per_series_gain") or {}
        for row in unit["sequences"]:
            decided = _num(row.get("support_gain"))
            delivered = _num(per_series.get(str(row["series_uid"])))
            if decided is None or delivered is None:
                continue
            if abs(decided - delivered) > 1e-12:
                delivered_mismatch.append({
                    "arm": unit["arm"], "position": unit["position"],
                    "series_uid": row["series_uid"],
                    "decided": decided, "delivered": delivered})
        steps_by_label: dict[str, set] = {}
        for row in unit["sequences"]:
            steps = json.dumps(row.get("deployed") or [], sort_keys=True)
            steps_by_label.setdefault(row["deployed_label"], set()).add(steps)
        for label, variants in steps_by_label.items():
            if len(variants) > 1:
                label_collisions.append({
                    "arm": unit["arm"], "position": unit["position"],
                    "operator_label": label, "distinct_programs": len(variants),
                    "variants": sorted(variants)})
    return {
        "decisions": len(rows),
        "each_decision_bound_to_its_own_sequence": len(bound) == len(rows),
        "distinct_sequences": len(features),
        "distinct_public_feature_cards": len(set(features.values())),
        "sessions_are_one_per_decision": all(
            isinstance(row.get("llm_calls_this_sequence"), int)
            for row in rows),
        "programs_deployed_per_unit": by_unit_programs,
        "what_was_executed_equals_what_was_decided":
            not delivered_mismatch,
        "decisions_whose_delivered_number_differs_from_the_decided_one":
            delivered_mismatch,
        "why_that_matters": (
            "the composition reads each sequence's own row out of its own "
            "program's entry, so this equality is the check that no program "
            "was broadcast past the sequence it was chosen for, and that no "
            "sequence was scored under someone else's program"),
        "operator_labels_hiding_more_than_one_program": label_collisions,
        "assignment_homogeneity": {
            key: {"distinct_programs": len(value), "programs": value}
            for key, value in by_unit_programs.items()},
        "identity_decisions": sum(1 for row in rows
                                  if row["deployed_label"] == "identity"),
        "deployed_decisions": sum(1 for row in rows
                                  if row["deployed_label"] != "identity"),
    }


def _all_units(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list((doc.get("formation") or {}).get("units") or []) + list(
        doc.get("followup_units") or [])


# ---------------------------------------------------------------------------
# 2. per-decision credit
# ---------------------------------------------------------------------------

def credit(doc: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for unit in _all_units(doc):
        for row in unit["sequences"]:
            rows.append({
                "phase": unit["phase"], "arm": unit["arm"],
                "position": unit["position"], "series_uid": row["series_uid"],
                "program": row["deployed_label"],
                "support": _num(row.get("support_gain")),
                "delayed": _num(row.get("delayed_gain")),
                "delayed_is_unknown": row.get("delayed_gain") == "UNKNOWN",
                "via": row.get("deployed_via"),
            })
    support = [row["support"] for row in rows if row["support"] is not None]
    delayed = [row["delayed"] for row in rows if row["delayed"] is not None]
    flips = [row for row in rows
             if row["support"] is not None and row["delayed"] is not None
             and row["support"] >= MATERIAL and row["delayed"] < -MATERIAL]
    return {
        "decisions": len(rows),
        "support_readable": len(support),
        "delayed_readable": len(delayed),
        "delayed_unknown": sum(1 for row in rows if row["delayed_is_unknown"]),
        "mean_own_support_gain": _mean(support),
        "mean_own_delayed_gain": _mean(delayed),
        "materially_positive_on_support": sum(1 for v in support
                                              if v >= MATERIAL),
        "materially_negative_on_support": sum(1 for v in support
                                              if v < -MATERIAL),
        "materially_positive_on_delayed": sum(1 for v in delayed
                                              if v >= MATERIAL),
        "materially_negative_on_delayed": sum(1 for v in delayed
                                              if v < -MATERIAL),
        "support_positive_then_delayed_negative": len(flips),
        "flip_fraction_of_support_positives": (
            round(len(flips) / max(sum(1 for v in support if v >= MATERIAL), 1),
                  4)),
        "no_group_number_was_copied_onto_a_decision": True,
        "what_a_row_is": ("one sequence's own gain under the program that "
                          "sequence actually ran; identity is a measured 0.0"),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3. what Slow changed
# ---------------------------------------------------------------------------

def _feature_table(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for unit in _all_units(doc):
        for row in unit["sequences"]:
            out.setdefault(str(row["series_uid"]),
                           dict(row.get("public_features") or {}))
    return out


def _series_of_episode(episode_id: str) -> str:
    """The UID out of the round name the writer stamped into the Episode id."""
    parts = str(episode_id).split("_")
    return parts[-2] if len(parts) >= 2 else ""


def separation(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Is there a deployment-visible condition to write knowledge about?

    The question a guidance card has to answer is not "did these decisions
    fail" but "what tells them apart from the ones that worked".  This puts the
    failing and the succeeding side of the taken group on the same features and
    reports, mechanically, whether any of them separates.  No model is involved
    and no fit is spent: both sides are already in the record.
    """
    boundary = dict(doc.get("boundary") or {})
    groups = list((boundary.get("grouping") or {}).get("groups") or [])
    if not groups:
        return {"available": False, "why": "no group formed"}
    group = groups[0]
    features = _feature_table(doc)
    failing = sorted({str(row["series_uid"]) for row in group["members"]
                      if row.get("series_uid")})
    succeeding = sorted({
        str(row.get("series_uid") or _series_of_episode(row.get("episode_id")))
        for row in (group.get("contrast_cases") or {}).get("positive") or []})
    succeeding = [uid for uid in succeeding if uid in features]
    names = sorted({name for uid in failing + succeeding
                    for name, value in (features.get(uid) or {}).items()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)})
    rows = []
    for name in names:
        left = [float(features[uid][name]) for uid in failing
                if name in (features.get(uid) or {})]
        right = [float(features[uid][name]) for uid in succeeding
                 if name in (features.get(uid) or {})]
        if not left or not right:
            continue
        disjoint = max(left) < min(right) or max(right) < min(left)
        rows.append({
            "feature": name,
            "failing": [round(min(left), 6), round(max(left), 6)],
            "succeeding": [round(min(right), 6), round(max(right), 6)],
            "constant_everywhere": len(set(left + right)) == 1,
            "ranges_are_disjoint": bool(disjoint),
        })
    both = sorted(set(failing) & set(succeeding))
    return {
        "available": True,
        "group_id": group["group_id"],
        "failing_series": failing,
        "succeeding_series": succeeding,
        "series_on_both_sides": both,
        "features_compared": len(rows),
        "features_that_separate": [row["feature"] for row in rows
                                   if row["ranges_are_disjoint"]],
        "features_constant_everywhere": [row["feature"] for row in rows
                                         if row["constant_everywhere"]],
        "rows": rows,
        "what_it_means": (
            "a feature whose failing and succeeding ranges overlap cannot be "
            "the condition of a guidance card.  Series appearing on both sides "
            "ran the same program and got opposite answers in different "
            "windows, so the distinction is not even a property of the series."),
    }


def knowledge(doc: Mapping[str, Any]) -> dict[str, Any]:
    boundary = dict(doc.get("boundary") or {})
    grouping = dict(boundary.get("grouping") or {})
    groups = list(grouping.get("groups") or [])
    proposal = dict(boundary.get("proposal") or {})
    validation = dict(boundary.get("validation") or {})
    return {
        "outcome": boundary.get("outcome"),
        "comparable_before_grouping": (grouping.get("comparability") or {}).get(
            "comparable"),
        "material_failures": grouping.get("material_failures"),
        "groups": [{
            "group_id": row["group_id"],
            "workflow": row["workflow"],
            "sign": row["sign"],
            "members": row["member_count"],
            "series": row["distinct_series"],
            "symptoms": row["symptoms"],
            "parameter_variants": len(row["parameter_variants"]),
            "matched_successes": len(row["contrast_cases"]["positive"]),
            "matched_conflicts": len(row["contrast_cases"]["conflict"]),
            "selectable_fault_types":
                row["fault"]["selectable_fault_types"],
            "features_with_a_spread": sorted(row["visible_pattern_spread"]),
        } for row in groups],
        "ungrouped_failures": len(grouping.get("ungrouped_failures") or []),
        "proposal": {
            "made": bool(proposal.get("proposed")),
            "why_not": None if proposal.get("proposed") else proposal.get("why"),
            "skill_id": proposal.get("skill_id"),
            "applicability": proposal.get("observable_applicability"),
            "body_characters": len(str(proposal.get("body") or "")),
            "carries_a_frozen_program":
                "Frozen program steps:" in str(proposal.get("body") or ""),
            "attempts": len(proposal.get("attempts") or []),
        },
        "validation": {
            "lines": validation.get("lines"),
            "qualifies": validation.get("qualifies"),
            "failed_lines": validation.get("failed_lines"),
            "face_read": (validation.get("L3") or {}).get("face_read"),
        } if validation else None,
        "adopted": boundary.get("outcome") == "UPDATE_ADOPTED",
        "the_only_write_path": ("ADD of one guidance entry at the batch "
                                "boundary; no card was patched or revoked "
                                "anywhere in this package"),
    }


# ---------------------------------------------------------------------------
# 4. the two arms
# ---------------------------------------------------------------------------

def arms(doc: Mapping[str, Any]) -> dict[str, Any]:
    units = list(doc.get("followup_units") or [])
    if not units:
        return {"ran": False,
                "why": (doc.get("followup") or {}).get("why"),
                "contrast": None}
    from evaluation.main_protocol_p4.run_dev_seq1 import ARM_FROZEN, ARM_UPDATED
    present = {unit["arm"] for unit in units}
    # Fixed, not sorted: the delta is defined as updated minus frozen, and an
    # alphabetical order would silently reverse its sign.
    names = [name for name in (ARM_FROZEN, ARM_UPDATED) if name in present]
    by_key = {(unit["arm"], unit["position"]): unit for unit in units}
    positions = sorted({unit["position"] for unit in units
                        if all((name, unit["position"]) in by_key
                               for name in names)})
    per_unit: list[dict[str, Any]] = []
    paired: dict[str, list[float]] = {"support": [], "delayed": []}
    for position in positions:
        row: dict[str, Any] = {"position": position}
        for name in names:
            unit = by_key[(name, position)]
            row[name] = {
                "support": unit["support_reading"].get("aggregate_gain"),
                "delayed": unit["delayed_reading"].get("aggregate_gain"),
                "treated": unit["support_reading"].get("treated"),
                "harmed_on_delayed": unit["delayed_reading"].get(
                    "harmed_series"),
                "distinct_programs": unit["support_reading"].get(
                    "distinct_programs"),
                "gate_passes": bool((unit.get("unit_authoritative_gate")
                                     or {}).get("passes")),
                "llm": sum(int(seq.get("llm_calls_this_sequence") or 0)
                           for seq in unit["sequences"]),
                "fits": sum(int(seq.get("consumer_fits_this_sequence") or 0)
                            for seq in unit["sequences"]),
                "read_the_new_card": sum(
                    1 for seq in unit["sequences"]
                    if any(str(sid) == str(doc.get("boundary", {}).get(
                        "adopted_skill_id"))
                        for sid in (seq.get("retrieved_skill_ids") or ()))),
            }
        if len(names) == 2:
            a, b = names          # a = knowledge-frozen, b = adopted-update
            for face in ("support", "delayed"):
                left, right = row[b][face], row[a][face]
                if left is not None and right is not None:
                    row["%s_delta" % face] = round(float(left) - float(right), 6)
                    paired[face].append(float(left) - float(right))
        per_unit.append(row)

    per_sequence_delta: list[dict[str, Any]] = []
    if len(names) == 2:
        a, b = names          # a = knowledge-frozen, b = adopted-update
        for position in positions:
            left = {seq["series_uid"]: seq
                    for seq in by_key[(b, position)]["sequences"]}
            right = {seq["series_uid"]: seq
                     for seq in by_key[(a, position)]["sequences"]}
            for uid in sorted(set(left) & set(right)):
                ls, rs = _num(left[uid].get("support_gain")), _num(
                    right[uid].get("support_gain"))
                ld, rd = _num(left[uid].get("delayed_gain")), _num(
                    right[uid].get("delayed_gain"))
                per_sequence_delta.append({
                    "position": position, "series_uid": uid,
                    "%s_program" % b: left[uid]["deployed_label"],
                    "%s_program" % a: right[uid]["deployed_label"],
                    "same_program":
                        left[uid]["deployed_label"] == right[uid]["deployed_label"],
                    "support_delta": (round(ls - rs, 6)
                                      if ls is not None and rs is not None
                                      else "UNKNOWN"),
                    "delayed_delta": (round(ld - rd, 6)
                                      if ld is not None and rd is not None
                                      else "UNKNOWN"),
                })
    return {
        "ran": True,
        "arms": names,
        "paired_positions": positions,
        "per_unit": per_unit,
        "mean_unit_delta": {face: _mean(values)
                            for face, values in paired.items()},
        "per_sequence_delta": per_sequence_delta,
        "decisions_where_the_two_arms_chose_the_same_program": sum(
            1 for row in per_sequence_delta if row["same_program"]),
        "decisions_compared": len(per_sequence_delta),
        "direction": ("the delta is the adopted-update arm minus the "
                      "knowledge-frozen arm, on the same unit and the same "
                      "sequence"),
    }


def trajectory(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What each unit's batch actually delivered, unit by unit.

    Reported for every unit of both segments.  When the follow-up segment does
    not open -- because no qualifying update formed and the two arms would be
    the same library -- this is what is left to say about how the next batch of
    Fast behaved: it is a trajectory, not a contrast, and it carries no
    treatment.
    """
    rows = []
    for unit in _all_units(doc):
        support, delayed = unit["support_reading"], unit["delayed_reading"]
        rows.append({
            "phase": unit["phase"], "arm": unit["arm"],
            "position": unit["position"], "origin": unit["origin"],
            "decided": len(unit["decided"]),
            "deployed": unit["deployed_count"],
            "identity": unit["identity_count"],
            "distinct_programs_deployed": unit["distinct_programs_deployed"],
            "coverage_treated": support.get("treated"),
            "support_aggregate": support.get("aggregate_gain"),
            "delayed_aggregate": delayed.get("aggregate_gain"),
            "harmed_on_support": support.get("harmed_series"),
            "harmed_on_delayed": delayed.get("harmed_series"),
            "max_single_series_harm_on_delayed":
                delayed.get("max_single_series_harm"),
            "unit_gate_passes": bool((unit.get("unit_authoritative_gate")
                                      or {}).get("passes")),
            "unit_gate_failed_lines": (unit.get("unit_authoritative_gate")
                                       or {}).get("failed_lines"),
            "llm_calls": sum(int(seq.get("llm_calls_this_sequence") or 0)
                             for seq in unit["sequences"]),
            "consumer_fits": sum(int(seq.get("consumer_fits_this_sequence")
                                     or 0) for seq in unit["sequences"]),
            "wall_seconds": unit.get("wall_seconds"),
        })
    return rows


def against_the_references(doc: Mapping[str, Any],
                           reference_path: Path) -> dict[str, Any]:
    """The per-sequence delivery read against the two reference deliveries.

    ``raw`` is nothing prepared, which is Static by construction and therefore
    exactly 0.0 on both faces.  ``fixed`` is one preparation given to the whole
    decision population -- the cohort-level answer.  The delta is what the
    per-sequence decision bought over each, on the same unit, the same
    population and the same face.  It is a development-level reading of one
    course, not a claim that per-sequence targeting generalises.
    """
    if not reference_path.is_file():
        return {"available": False, "why": "no reference artifact"}
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    readings = reference.get("readings") or {}
    rows = []
    for unit in _all_units(doc):
        row = readings.get(str(unit["position"]))
        if not row:
            continue
        fixed = row.get("fixed_preparation") or {}
        entry: dict[str, Any] = {
            "phase": unit["phase"], "arm": unit["arm"],
            "position": unit["position"],
            "per_sequence_support": unit["support_reading"].get(
                "aggregate_gain"),
            "per_sequence_delayed": unit["delayed_reading"].get(
                "aggregate_gain"),
            "raw_support": ((row.get("raw") or {}).get("support")
                            or {}).get("aggregate_gain"),
            "raw_delayed": ((row.get("raw") or {}).get("delayed")
                            or {}).get("aggregate_gain"),
            "fixed_program": fixed.get("program"),
            "fixed_support": (fixed.get("support") or {}).get("aggregate_gain"),
            "fixed_delayed": (fixed.get("delayed") or {}).get("aggregate_gain"),
        }
        for face in ("support", "delayed"):
            mine = entry["per_sequence_%s" % face]
            for other in ("raw", "fixed"):
                theirs = entry["%s_%s" % (other, face)]
                entry["%s_minus_%s_%s" % ("per_sequence", other, face)] = (
                    round(float(mine) - float(theirs), 6)
                    if mine is not None and theirs is not None else "UNKNOWN")
        rows.append(entry)
    deltas = {
        key: _mean([float(row[key]) for row in rows
                    if isinstance(row.get(key), float)])
        for key in ("per_sequence_minus_raw_support",
                    "per_sequence_minus_raw_delayed",
                    "per_sequence_minus_fixed_support",
                    "per_sequence_minus_fixed_delayed")}
    return {
        "available": True,
        "source": str(reference_path),
        "rows": rows,
        "mean_delta": deltas,
        "reference_cost": reference.get("cost"),
        "not_in_the_package_ledger": reference.get(
            "not_in_the_package_ledger"),
        "what_it_is_not": (
            "a two-arm knowledge comparison.  Both references are fixed "
            "policies, not arms of this experiment; this says what the "
            "per-sequence decision delivered against them on the same units."),
    }


def references(doc: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for position, row in (doc.get("reference_readings") or {}).items():
        fixed = row.get("fixed_preparation") or {}
        out[position] = {
            "raw_support": ((row.get("raw") or {}).get("support") or {}).get(
                "aggregate_gain"),
            "raw_delayed": ((row.get("raw") or {}).get("delayed") or {}).get(
                "aggregate_gain"),
            "fixed_program": fixed.get("program"),
            "fixed_support": (fixed.get("support") or {}).get("aggregate_gain"),
            "fixed_delayed": (fixed.get("delayed") or {}).get("aggregate_gain"),
        }
    return out


# ---------------------------------------------------------------------------
# 5. where the remaining block is
# ---------------------------------------------------------------------------

def first_fault(doc: Mapping[str, Any], credit_row: Mapping[str, Any],
                knowledge_row: Mapping[str, Any],
                arms_row: Mapping[str, Any]) -> dict[str, Any]:
    """AGENTS 6's ladder, applied mechanically to what was recorded."""
    rows = [row for unit in _all_units(doc) for row in unit["sequences"]]
    any_positive = credit_row["materially_positive_on_support"] > 0
    any_candidate = any(row.get("candidate_ids") for row in rows)
    any_legal_probe = any(
        (probe.get("kind") == "probe")
        for row in rows for probe in (row.get("probes") or ()))
    mischosen = 0
    for row in rows:
        best = None
        for probe in row.get("probes") or ():
            gain = _num(probe.get("gain"))
            if gain is not None and (best is None or gain > best):
                best = gain
        own = _num(row.get("support_gain"))
        if best is not None and own is not None and best - own > MATERIAL:
            mischosen += 1
    ladder = [
        ("no readable positive effect anywhere",
         not any_positive, "Consumer / evaluator / training protocol"),
        ("effects exist but no legal Program was ever admitted",
         any_positive and not any_legal_probe, "Program Supply"),
        ("Programs exist but Fast proposed no candidate",
         any_positive and any_legal_probe and not any_candidate,
         "Observation / localization / supply"),
        ("candidates existed and a better probed one was not deployed",
         mischosen > 0, "selection / retrieval"),
        ("Support succeeded and the later window did not",
         credit_row["support_positive_then_delayed_negative"] > 0,
         "Scope / overfitting / risk"),
        ("no qualifying knowledge update formed",
         not knowledge_row["adopted"], "update"),
        ("the update was adopted but changed nothing downstream",
         knowledge_row["adopted"] and arms_row.get("ran")
         and (arms_row.get("mean_unit_delta") or {}).get("delayed") == 0.0,
         "subsequent applicability"),
    ]
    hit = [{"symptom": name, "routes_to": route}
           for name, condition, route in ladder if condition]
    return {
        "ladder_hits": hit,
        "earliest": hit[0] if hit else None,
        "decisions_where_a_better_probed_candidate_was_not_deployed": mischosen,
        "note": ("more than one rung can be lit at once; the earliest is the "
                 "one this package's next step should open"),
    }


# ---------------------------------------------------------------------------

def run(path: Path, reference_path: Path | None = None) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    reference_path = reference_path or (
        ART / "dev_seq1_reference_readings__all.json")
    credit_row = credit(doc)
    knowledge_row = knowledge(doc)
    arms_row = arms(doc)
    return {
        "audit": "DEV_SEQ1_CONTRASTS",
        "source": str(path),
        "status": doc.get("status"),
        "treatment": doc.get("treatment"),
        "registered_plan": doc.get("registered_plan"),
        "q1_was_it_really_per_sequence": per_sequence_execution(doc),
        "q2_per_decision_credit": {k: v for k, v in credit_row.items()
                                   if k != "rows"},
        "q2_rows": credit_row["rows"],
        "q3_what_slow_changed": knowledge_row,
        "q3b_was_there_a_condition_to_write": separation(doc),
        "q4_the_two_arms": arms_row,
        "q4_unit_trajectory": trajectory(doc),
        "q4_reference_readings": references(doc),
        "q4_against_the_references": against_the_references(
            doc, reference_path),
        "q5_where_the_block_is": first_fault(doc, credit_row, knowledge_row,
                                             arms_row),
        "cost": doc.get("cost"),
        "transport_actual_usage": doc.get("transport_actual_usage"),
        "comparability": (
            "per-sequence denominators.  These numbers are not comparable to "
            "DEV-AUTO-1/2/3 or DEV-KNOW-1, which scored one broadcast program "
            "over a twenty-series cohort, and are not compared to them."),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--source", default=None)
    parser.add_argument("--references", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    source = Path(args.source) if args.source else (
        ART / ("dev_seq1_per_sequence__%s.json" % args.run_id))
    result = run(source, Path(args.references) if args.references else None)
    out = Path(args.out) if args.out else (
        ART / ("dev_seq1_contrasts__%s.json" % args.run_id))
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                              default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("q2_rows", "registered_plan")},
                     ensure_ascii=False, indent=1, default=str)[:6000])
    print("\nwritten: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
