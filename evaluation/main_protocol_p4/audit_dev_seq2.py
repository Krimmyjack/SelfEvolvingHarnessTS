"""DEV-SEQ-2 contrasts: the six questions the report has to answer.

Offline.  Zero Consumer fits, zero LLM calls -- everything here is read out of
the two group artifacts and the DEV-SEQ-1 receipts, which are not modified.

    1  are the two engineering fixes really wired in, and what model ran?
    2  what did Slow see, what did it change, why, and what is still a guess?
    3  did the new guidance change Fast's observation, candidates, selection
       and actual delivery -- or only what was loaded?
    4  the validation unit and the follow-up units: utility, harm, coverage,
       cost, with the adopted and unadopted branches kept apart;
    5  do the two groups agree, and if not, is that no treatment, a different
       edit, or the same edit landing differently?
    6  what is the largest thing still unresolved?

The behaviour chain is reported as five separate links, because they fail
separately: loaded -> observation -> candidates -> selection/delivery ->
effect.  Being retrieved is not the same as being used, a text difference is
not a capability difference, and a difference in score with no difference in
behaviour is sampling noise, not learning.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_seq2_knowledge as know
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_seq2 as run
from SelfEvolvingHarnessTS.methods.ttha.retrieval import evaluate_applicability

ART = base.ROOT / "artifacts" / "main_protocol"
MATERIAL = know.MATERIAL


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _by_arm(units: Sequence[Mapping[str, Any]], position: int
            ) -> dict[str, Mapping[str, Any]]:
    return {row["arm"]: row for row in units if int(row["position"]) == position}


# ---------------------------------------------------------------------------
# Q1  the two fixes, and the model
# ---------------------------------------------------------------------------

def fixes_and_model(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, doc in groups.items():
        binding = doc.get("feature_binding") or {}
        ledger = doc.get("measurement_ledger") or {}
        usage = doc.get("transport_actual_usage") or {}
        rows[name] = {
            "window_binding": {
                "keyed_by": binding.get("keyed_by"),
                "windows": binding.get("windows"),
                "distinct_series": binding.get("distinct_series"),
                "windows_per_series": (
                    round(binding["windows"] / binding["distinct_series"], 2)
                    if binding.get("distinct_series") else None),
                "misses": len(binding.get("misses") or []),
                "in_force": binding.get("keyed_by") == "(series_uid, origin)",
            },
            "measurement_dedup": {
                "references": ledger.get("references"),
                "distinct_measurements": ledger.get("distinct_measurements"),
                "repeated_references": ledger.get("repeated_references"),
                "distinct_series": ledger.get("distinct_series"),
                "distinct_windows": ledger.get("distinct_windows"),
                "in_force": ledger.get("distinct_measurements") is not None,
            },
            "model": {
                "requested": usage.get("requested_model"),
                "returned": usage.get("returned_models"),
                "one_model_only": (len(usage.get("returned_models") or []) == 1),
                "is_the_fixed_non_flash_model": (
                    usage.get("returned_models") == [run.REQUIRED_MODEL]),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        }
    return {
        "per_group": rows,
        "what_changed_from_dev_seq_1": {
            "window_binding": (
                "DEV-SEQ-1 keyed a visible-feature card by series UID and kept "
                "the first construction window's card, so every u8 Episode was "
                "described to Slow, and scored by the validator, with u7's "
                "features.  A card is now keyed by (series_uid, origin) and a "
                "missing window is a recorded miss, never a borrowed card"),
            "measurement_dedup": (
                "DEV-SEQ-1's L3 iterated group members, so one real reading "
                "cited by two Episodes of the same series and program counted "
                "twice.  A measurement is now keyed by the prediction cache's "
                "own key plus the served UID; no Episode is deleted and a "
                "different window, program, parameter set or training config "
                "stays a separate measurement"),
            "model": (
                "DEV-SEQ-1 requested deepseek-chat, which this relay answers "
                "as deepseek-v4-flash.  This package fixes on %s and checks "
                "the returned identity, not the requested alias.  That makes "
                "it a different experiment from DEV-SEQ-1 and the two are not "
                "tabled together" % run.REQUIRED_MODEL),
        },
    }


def dev_seq1_l2_recomputed() -> dict[str, Any]:
    """DEV-SEQ-1's refusal, recomputed with correct window binding.

    A fix is not a reason to approve the old card.  This says only whether
    the old refusal was an artefact of the mis-binding or would stand anyway.
    """
    doc = _load(ART / "dev_seq1_per_sequence__run2.json")
    if doc is None:
        return {"available": False}
    boundary = doc.get("boundary") or {}
    proposal = boundary.get("proposal") or {}
    applicability = proposal.get("observable_applicability") or {}
    groups = (boundary.get("grouping") or {}).get("groups") or []
    if not applicability or not groups:
        return {"available": False, "why": "no predicate or no group in run2"}
    group = groups[0]

    window_features: dict[tuple[str, int], dict[str, Any]] = {}
    for unit in (doc.get("formation") or {}).get("units") or ():
        for row in unit.get("sequences") or ():
            window_features[(str(row["series_uid"]), int(row["origin"]))] = dict(
                row.get("public_features") or {})

    def matches(features: Mapping[str, Any]) -> bool:
        try:
            return bool(evaluate_applicability(applicability, features)[0])
        except Exception:  # noqa: BLE001
            return False

    windows = sorted({(str(m["series_uid"]), int(m["origin"]))
                      for m in group["members"]})
    covered = [key for key in windows if matches(window_features.get(key, {}))]
    old = boundary.get("validation", {}).get("L2") or {}
    return {
        "available": True,
        "predicate": applicability,
        "old_reading": {"failing_series": old.get("failing_series"),
                        "covered": old.get("covered"),
                        "verdict": "L2 failed: 1 of 6 series"},
        "recomputed_with_window_binding": {
            "failing_windows": [list(key) for key in windows],
            "covered": [list(key) for key in covered],
            "covered_fraction": (round(len(covered) / len(windows), 4)
                                 if windows else None),
            "L2_would_pass": len(covered) * 2 > len(windows),
        },
        "verdict": (
            "the refusal was not an artefact of the mis-binding: with each "
            "failing decision tested against its own window's features the "
            "predicate still covers %d of %d, so L2 still fails.  This is "
            "reported to separate 'the old L2 rejected it' from 'there was no "
            "useful condition'; it is not a reason to approve the old card, "
            "which stays unadopted and is not re-run here"
            % (len(covered), len(windows))),
    }


# ---------------------------------------------------------------------------
# Q2  what Slow saw and changed
# ---------------------------------------------------------------------------

def what_slow_did(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, doc in groups.items():
        boundary = doc.get("boundary") or {}
        proposal = boundary.get("proposal") or {}
        grouping = boundary.get("grouping") or {}
        group = (grouping.get("groups") or [{}])[0]
        applied = boundary.get("applied") or {}
        rows[name] = {
            "outcome": boundary.get("outcome"),
            "what_it_was_shown": {
                "episodes": boundary.get("episodes_in_the_slow_input"),
                "material_failures": grouping.get("material_failures"),
                "groups_formed": grouping.get("group_count"),
                "group_taken": boundary.get("group_taken"),
                "members": group.get("member_count"),
                "distinct_series": len(group.get("distinct_series") or []),
                "distinct_windows": len(group.get("distinct_windows") or []),
                "matched_successes": len(
                    (group.get("contrast_cases") or {}).get("positive") or []),
                "ungrouped_failures_kept_as_episodes": len(
                    grouping.get("ungrouped_failures") or []),
                "surfaces_offered": boundary.get("surfaces_offered"),
                "fault_labels_selectable": (group.get("fault") or {}).get(
                    "selectable_fault_types"),
            },
            "what_it_changed": {
                "surface": proposal.get("target_surface_id"),
                "operation": proposal.get("operation"),
                "derived_cause": proposal.get("derived_cause"),
                "compiled": applied.get("applied"),
                "parent_sha": applied.get("parent_runtime_bundle_sha"),
                "candidate_sha": applied.get("runtime_bundle_sha"),
                "text": proposal.get("written_text"),
                "applicability": proposal.get("observable_applicability"),
                "attempts": proposal.get("attempts"),
                "why_no_proposal": (proposal.get("detail")
                                    if not proposal.get("proposed") else None),
            },
            "what_it_predicted": {
                "behaviour": proposal.get("predicted_agent_behavior_change"),
                "effect": proposal.get("predicted_data_effect"),
                "falsified_by": proposal.get("falsification_condition"),
            },
            "what_is_still_a_guess": (
                "the mechanism.  The card carries facts -- programs, windows, "
                "both faces, the visible features of failures and of matched "
                "successes -- and the edit is one provisional account of them. "
                "Nothing in this package establishes that account is the cause; "
                "only the two arms say whether acting on it helped"),
            "coverage_diagnostic_not_a_gate": boundary.get(
                "legacy_lines_diagnostic"),
        }
    return rows


# ---------------------------------------------------------------------------
# Q3  the behaviour chain
# ---------------------------------------------------------------------------

def _pair_rows(unit_a: Mapping[str, Any], unit_b: Mapping[str, Any]
               ) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    a = {str(row["series_uid"]): row for row in unit_a.get("sequences") or ()}
    b = {str(row["series_uid"]): row for row in unit_b.get("sequences") or ()}
    return [(a[uid], b[uid]) for uid in sorted(set(a) & set(b))]


def behaviour_chain(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Five links, counted on the pairs the edit could actually reach.

    Counting them over the whole population is wrong when the edit reaches
    only part of it: a pair whose two arms hold the same effective knowledge
    can still differ in observation, pool, delivery and score, because the
    model samples.  Those differences are not the edit.  So the links are
    computed on the **exposed** pairs, and the same links on the unexposed
    pairs are reported beside them as what sampling alone produces in this
    very run.
    """
    exposure = (doc.get("boundary") or {}).get("exposure") or {}
    reached = {(str(row["position"]), str(row["series_uid"]))
               for row in exposure.get("per_decision") or ()
               if row.get("edit_is_in_the_candidate_view")}
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    positions: list[int] = []
    validation = ((doc.get("validation") or {}).get("units") or {})
    if run.ARM_A in validation and run.ARM_B in validation:
        pairs += _pair_rows(validation[run.ARM_A], validation[run.ARM_B])
        positions.append(run.VALIDATION_POSITION)
    for position in run.FOLLOWUP_POSITIONS:
        arms = _by_arm(doc.get("followup_units") or (), position)
        if run.ARM_A in arms and run.ARM_B in arms:
            pairs += _pair_rows(arms[run.ARM_A], arms[run.ARM_B])
            positions.append(position)
    if not pairs:
        return {"paired_decisions": 0,
                "why": "no unit was run by both arms"}

    def differs(a: Any, b: Any) -> bool:
        return json.dumps(a, sort_keys=True, default=str) != json.dumps(
            b, sort_keys=True, default=str)

    def links(subset: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
              ) -> dict[str, Any]:
        effect = [
            (a["series_uid"], x, y) for a, b, x, y in (
                (a, b, _num(a.get("delayed_gain")), _num(b.get("delayed_gain")))
                for a, b in subset)
            if x is not None and y is not None and abs(y - x) >= MATERIAL]
        return {
            "pairs": len(subset),
            "link_1_the_edit_is_loaded": {
                "in_arm_b": sum(1 for _a, b in subset
                                if b.get("edited_text_is_in_this_sequences_view")),
                "in_arm_a": sum(1 for a, _b in subset
                                if a.get("edited_text_is_in_this_sequences_view")),
            },
            "link_1b_retrieved_skill_set_differs": sum(
                1 for a, b in subset
                if differs(sorted(a.get("retrieved_skill_ids") or []),
                           sorted(b.get("retrieved_skill_ids") or []))),
            "link_2_observation_differs": sum(
                1 for a, b in subset
                if differs(a.get("inspected_regions"), b.get("inspected_regions"))
                or a.get("tool_calls") != b.get("tool_calls")),
            "link_3_candidate_pool_differs": sum(
                1 for a, b in subset
                if differs(sorted((a.get("candidate_programs") or {}).values()),
                           sorted((b.get("candidate_programs") or {}).values()))),
            "link_4_selection_or_delivery_differs": {
                "different_program_delivered": sum(
                    1 for a, b in subset
                    if a.get("deployed_label") != b.get("deployed_label")),
                "different_decision_kind_deploy_vs_identity": sum(
                    1 for a, b in subset
                    if (a.get("decision") == "DEPLOYED")
                    != (b.get("decision") == "DEPLOYED")),
            },
            "link_5_material_effect_difference": {
                "count": len(effect),
                "cases": [{"series_uid": uid, "arm_a": x, "arm_b": y,
                           "difference": round(y - x, 6)}
                          for uid, x, y in effect],
            },
        }

    def is_exposed(a: Mapping[str, Any]) -> bool:
        return (str(a.get("position")), str(a.get("series_uid"))) in reached

    exposed = [(a, b) for a, b in pairs if is_exposed(a)]
    unexposed = [(a, b) for a, b in pairs if not is_exposed(a)]
    return {
        "positions_compared": positions,
        "paired_decisions": len(pairs),
        "on_the_pairs_the_edit_reached": links(exposed),
        "on_the_pairs_it_did_not_reach": links(unexposed),
        "how_to_read_it": (
            "only the first block can carry a knowledge effect: those are the "
            "pairs whose two arms held different effective knowledge.  The "
            "second block is the same five links on pairs whose arms held the "
            "SAME effective knowledge, so every difference in it is sampling. "
            "Reading the two together, or reading the second as an effect, is "
            "the mistake this split exists to prevent.  Within the first "
            "block the links still fail separately: loaded with no downstream "
            "difference means the text was read and not used."),
    }


# ---------------------------------------------------------------------------
# Q4  utility, harm, coverage, cost
# ---------------------------------------------------------------------------

def _unit_face(unit: Mapping[str, Any]) -> dict[str, Any]:
    delayed = unit.get("delayed_reading") or {}
    support = unit.get("support_reading") or {}
    gate = unit.get("unit_authoritative_gate") or {}
    return {
        "arm": unit.get("arm"),
        "decided": len(unit.get("decided") or []),
        "deployed": unit.get("deployed_count"),
        "identity": unit.get("identity_count"),
        "distinct_programs": unit.get("distinct_programs_deployed"),
        "faulted": unit.get("faulted_sequences"),
        "support_aggregate": support.get("aggregate_gain"),
        "delayed_aggregate": delayed.get("aggregate_gain"),
        "treated": delayed.get("treated"),
        "harmed_fraction": delayed.get("harmed_fraction"),
        "max_single_series_harm": delayed.get("max_single_series_harm"),
        "authoritative_gate_passes": gate.get("passes") if gate else None,
        "wall_seconds": unit.get("wall_seconds"),
    }


def outcomes(doc: Mapping[str, Any]) -> dict[str, Any]:
    per_position: dict[str, Any] = {}
    validation = ((doc.get("validation") or {}).get("units") or {})
    if validation:
        per_position[str(run.VALIDATION_POSITION)] = {
            "phase": "validation",
            "arms": {name: _unit_face(unit) for name, unit in validation.items()},
            "paired_delayed": (
                know.paired_delayed(validation[run.ARM_A]["sequences"],
                                    validation[run.ARM_B]["sequences"])
                if run.ARM_A in validation and run.ARM_B in validation else None),
        }
    per_unit_differences: list[float] = []
    incomplete: list[str] = []
    for position in run.FOLLOWUP_POSITIONS:
        arms = _by_arm(doc.get("followup_units") or (), position)
        if not arms:
            continue
        pair = (know.paired_delayed(arms[run.ARM_A]["sequences"],
                                    arms[run.ARM_B]["sequences"])
                if run.ARM_A in arms and run.ARM_B in arms else None)
        per_position[str(position)] = {
            "phase": "followup",
            "arms": {name: _unit_face(unit) for name, unit in arms.items()},
            "paired_delayed": pair,
            "reference": {
                key: {"delayed_aggregate": (value.get("delayed") or {}).get(
                    "aggregate_gain")}
                for key, value in ((doc.get("reference_readings") or {})
                                   .get(str(position)) or {}).items()
                if isinstance(value, Mapping)},
        }
        if pair is None:
            incomplete.append(str(position))
        elif isinstance(pair.get("whole_population_difference"), float):
            per_unit_differences.append(pair["whole_population_difference"])
        else:
            incomplete.append(str(position))

    followup_positions_run = [str(p) for p in run.FOLLOWUP_POSITIONS
                              if str(p) in per_position]
    main = {
        "rule": ("unit mean first, then equal weight over the registered "
                 "follow-up units; A and B on the same UID set and the same "
                 "denominator"),
        "followup_units_read": followup_positions_run,
        "per_unit_difference_b_minus_a": per_unit_differences,
        "main_readout": (_mean(per_unit_differences)
                         if per_unit_differences
                         and not incomplete else "UNKNOWN"),
        "units_with_a_missing_reading": incomplete,
        "readable_subset_only": (_mean(per_unit_differences)
                                 if per_unit_differences else "UNKNOWN"),
        "why_unknown": ("a missing required reading on either side makes the "
                        "whole-population difference UNKNOWN; the readable "
                        "subset is reported beside it, never in its place"),
    }
    adoption = doc.get("adoption") or {}
    branch = (doc.get("followup") or {}).get("branch_status")
    return {
        "per_position": per_position,
        "main_delayed_readout": main,
        "adoption": adoption,
        "branch_status": branch,
        "branch_meaning": (
            "ADOPTED_DEVELOPMENT_BRANCH: the candidate passed this package's "
            "development adoption mark on the validation unit, and its "
            "follow-up numbers are the numbers of an adopted development "
            "branch.  UNADOPTED_CANDIDATE_DIAGNOSTIC_BRANCH: it did not, and "
            "its follow-up numbers are candidate diagnostics -- never a "
            "promotion or deployment gain, and never a retroactive pass of "
            "the validation unit"),
        "cost": doc.get("cost"),
        "independence": {
            "distinct_series": (doc.get("measurement_ledger") or {}).get(
                "distinct_series"),
            "distinct_windows": (doc.get("measurement_ledger") or {}).get(
                "distinct_windows"),
            "how_to_read_n": (
                "rows are not independent samples.  The same series appears at "
                "several windows, one Consumer model per (unit, program) is "
                "shared by every series it serves, and both arms read the same "
                "deterministic predictions.  Count series and windows, not rows"),
        },
    }


def exposure_split(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The decisions the edit reached, against the ones it did not.

    When an edit reaches only part of the population, the population-wide
    paired difference is mostly a difference between two arms holding the
    *same* effective knowledge.  Those pairs are worth reporting -- they are
    an empirical spread measured inside this very run -- but they are not the
    knowledge effect, and the exposed pairs are too few to average.  So both
    are reported, separately, and neither is presented as the other.
    """
    exposure = (doc.get("boundary") or {}).get("exposure") or {}
    reached = {(str(row["position"]), str(row["series_uid"]))
               for row in exposure.get("per_decision") or ()
               if row.get("edit_is_in_the_candidate_view")}
    if not exposure:
        return {"available": False, "why": "no exposure receipt on this group"}

    units: list[tuple[int, Mapping[str, Any]]] = []
    validation = ((doc.get("validation") or {}).get("units") or {})
    if validation:
        units.append((run.VALIDATION_POSITION, validation))
    for position in run.FOLLOWUP_POSITIONS:
        arms = _by_arm(doc.get("followup_units") or (), position)
        if arms:
            units.append((position, arms))

    exposed_rows: list[dict[str, Any]] = []
    unexposed: list[float] = []
    unexposed_pairs = 0
    for position, arms in units:
        if run.ARM_A not in arms or run.ARM_B not in arms:
            continue
        a = {str(r["series_uid"]): r for r in arms[run.ARM_A]["sequences"]}
        b = {str(r["series_uid"]): r for r in arms[run.ARM_B]["sequences"]}
        for uid in sorted(set(a) & set(b)):
            x, y = _num(a[uid].get("delayed_gain")), _num(b[uid].get("delayed_gain"))
            hit = (str(position), uid) in reached
            if hit:
                exposed_rows.append({
                    "position": position, "series_uid": uid,
                    "arm_a_delayed": a[uid].get("delayed_gain"),
                    "arm_b_delayed": b[uid].get("delayed_gain"),
                    "difference": (round(y - x, 6)
                                   if x is not None and y is not None
                                   else "UNKNOWN"),
                    "arm_a_program": a[uid].get("deployed_label"),
                    "arm_b_program": b[uid].get("deployed_label"),
                    "delivered_differently": (a[uid].get("deployed_label")
                                              != b[uid].get("deployed_label")),
                    "arm_b_retrieved": b[uid].get("retrieved_skill_ids"),
                    "arm_a_action_facts": (a[uid].get("action_facts") or {}).get(
                        "support"),
                    "arm_b_action_facts": (b[uid].get("action_facts") or {}).get(
                        "support"),
                })
            elif x is not None and y is not None:
                unexposed.append(y - x)
                unexposed_pairs += 1

    spread = None
    if unexposed:
        values = sorted(unexposed)
        spread = {
            "pairs": unexposed_pairs,
            "mean": round(sum(values) / len(values), 6),
            "min": round(values[0], 6),
            "max": round(values[-1], 6),
            "identical_pairs": sum(1 for v in values if v == 0.0),
            "materially_different_pairs": sum(1 for v in values
                                              if abs(v) >= MATERIAL),
        }
    return {
        "available": True,
        "decisions_the_edit_reached": len(exposed_rows),
        "exposed_decisions": exposed_rows,
        "same_knowledge_pairs": spread,
        "how_to_read_this": (
            "the exposed rows are the only pairs where the two arms held "
            "different effective knowledge.  Every other pair ran under the "
            "same effective knowledge, so its difference is sampling: those "
            "are summarised as a spread measured inside this run.  It is "
            "descriptive, not a reliable noise floor, and the exposed rows "
            "are too few to average into a claim"),
    }


# ---------------------------------------------------------------------------
# Q5 / Q6
# ---------------------------------------------------------------------------

def consistency(groups: Mapping[str, Mapping[str, Any]],
                per_group: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for name, doc in groups.items():
        boundary = doc.get("boundary") or {}
        proposal = boundary.get("proposal") or {}
        rows[name] = {
            "treatment": doc.get("treatment"),
            "surface": proposal.get("target_surface_id"),
            "outcome": boundary.get("outcome"),
            "adoptable": (doc.get("adoption") or {}).get("adoptable"),
            "main_readout": (per_group[name]["outcomes"]
                             ["main_delayed_readout"]["main_readout"]),
            "behaviour_links_on_reached_pairs": (
                per_group[name]["behaviour_chain"].get(
                    "on_the_pairs_the_edit_reached")),
        }
    treatments = {row["treatment"] for row in rows.values()}
    surfaces = {row["surface"] for row in rows.values()}
    readouts = [row["main_readout"] for row in rows.values()]
    if len(treatments) > 1 or "NO_UPDATE_TREATMENT" in treatments:
        why = ("the groups differ in whether a candidate formed at all, so "
               "any difference between them is not a difference between two "
               "versions of the same edit")
    elif len(surfaces) > 1:
        why = ("the two groups edited different surfaces, so they are two "
               "different modifications rather than a repeat of one")
    else:
        why = ("the same surface in both groups, so a difference between them "
               "is a difference in what Fast decided, not in what Slow wrote")
    return {
        "per_group": rows,
        "agree_on_treatment": len(treatments) == 1,
        "agree_on_surface": len(surfaces) == 1,
        "main_readouts": readouts,
        "reading": why,
        "no_significance_is_claimed": (
            "two groups on the same data are random repeats, not independent "
            "replications; no test is run and no significance is claimed"),
    }


def open_question(per_group: Mapping[str, Any],
                  groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The earliest link in the chain that is actually blocked."""
    treatments = {doc.get("treatment") for doc in groups.values()}

    # A group whose edit compiled and then reached no decision is blocked at
    # loading, and that is a stronger statement than "no arm ran": the
    # candidate exists, it is legal, and it is invisible.
    never_exposed = {
        name: row["exposure"] for name, row in per_group.items()
        if isinstance(row.get("exposure"), Mapping)
        and row["exposure"].get("decisions_checked")
        and row["exposure"].get("decisions_the_edit_reaches") == 0}
    if never_exposed and all(doc.get("treatment") != "UPDATE_TREATMENT"
                             for doc in groups.values()):
        checked = sum(int(row.get("decisions_checked") or 0)
                      for row in never_exposed.values())
        return {
            "earliest_blocked_link": (
                "LOADING: the edit compiled into a legal candidate and then "
                "reached 0 of %d registered arm decisions, so no paired "
                "decision could differ.  The guidance's utility is UNTESTED, "
                "not measured as zero.  What is blocked is the step between "
                "'Slow can write a legal edit' and 'the edit is presented to "
                "the sequences it was written about'" % checked),
            "groups_never_exposed": sorted(never_exposed),
            "one_suggestion_only": (
                "one next step is named in the report.  No third group, no "
                "ablation and no sealed experiment is started from here"),
        }

    if treatments == {"NO_UPDATE_TREATMENT"}:
        rung = ("PROPOSAL: no legal candidate formed in either group, so "
                "nothing downstream was tested")
    else:
        chains = [row["behaviour_chain"]["on_the_pairs_the_edit_reached"]
                  for row in per_group.values()
                  if row["behaviour_chain"].get("paired_decisions")]
        loaded = sum((c["link_1_the_edit_is_loaded"]["in_arm_b"])
                     for c in chains)
        used = sum(c["link_2_observation_differs"]
                   + c["link_3_candidate_pool_differs"] for c in chains)
        delivered = sum(
            c["link_4_selection_or_delivery_differs"]["different_program_delivered"]
            for c in chains)
        helped = sum(c["link_5_material_effect_difference"]["count"]
                     for c in chains)
        if not chains:
            rung = "EXECUTION: no unit was run by both arms"
        elif loaded == 0:
            rung = ("LOADING: the edited text never reached the sequences Fast "
                    "decided for -- check the applicability condition and "
                    "retrieval top_k")
        elif used == 0 and delivered == 0:
            rung = ("USE: the edit was loaded into every decision and changed "
                    "neither the observation, the candidate pool nor the "
                    "delivered program.  Prose reached the prompt and did not "
                    "become behaviour")
        elif delivered == 0:
            rung = ("DELIVERY: the edit changed what was observed or proposed "
                    "but never what was delivered, so no reading could differ")
        elif helped == 0:
            rung = ("BENEFIT: behaviour changed and utility did not.  The "
                    "hypothesis Slow acted on is the thing to revise, not the "
                    "size of the edit menu")
        else:
            rung = ("FOLLOW-ON APPLICABILITY: behaviour changed and some "
                    "decisions improved; whether that survives new windows, "
                    "series and Tasks is untested here")
    return {
        "earliest_blocked_link": rung,
        "one_suggestion_only": (
            "one next step is named in the report.  No third group, no "
            "ablation and no sealed experiment is started from here"),
    }


# ---------------------------------------------------------------------------

def build(suffix: str = "", prefix: str = "dev_seq2_slow_update_to_fast"
          ) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for name in sorted(run.GROUPS):
        doc = _load(ART / ("%s__%s%s.json" % (prefix, name, suffix)))
        if doc is not None:
            groups[name] = doc
    if not groups:
        return {"stage": "DEV_SEQ2_CONTRASTS", "available": False,
                "why": "no group artifact found"}

    per_group = {
        name: {"status": doc.get("status"),
               "stopped_at": doc.get("stopped_at"),
               "treatment": doc.get("treatment"),
               "early_termination": doc.get("early_termination"),
               "exposure": {
                   k: v for k, v
                   in ((doc.get("boundary") or {}).get("exposure") or {}).items()
                   if k != "per_decision"},
               "behaviour_chain": behaviour_chain(doc),
               "exposure_split": exposure_split(doc),
               "outcomes": outcomes(doc)}
        for name, doc in groups.items()}

    # Groups that share a run id share one budget file, so each of their
    # ledgers already includes the earlier groups' spend -- summing them would
    # double-count.  Groups with different run ids have independent ledgers
    # and do add up.  (flash g1/g2 share "flash"; the sol groups used "pkg"
    # and "pkg2" and are independent.)
    by_run: dict[str, dict[str, float]] = {}
    for name, doc in groups.items():
        cost = doc.get("cost") or {}
        run_id = str(doc.get("run_id") or name)
        slot = by_run.setdefault(run_id, {"fits": 0.0, "llm": 0.0, "wall": 0.0})
        slot["fits"] = max(slot["fits"],
                           float(cost.get("physical_consumer_fits") or 0))
        slot["llm"] = max(slot["llm"], float(cost.get("llm_calls") or 0))
        slot["wall"] = max(slot["wall"], float(cost.get("wall_seconds") or 0.0))
    total_fits = int(sum(slot["fits"] for slot in by_run.values()))
    total_llm = int(sum(slot["llm"] for slot in by_run.values()))
    total_wall = sum(slot["wall"] for slot in by_run.values())
    return {
        "stage": ("DEV_SEQ3_CONTRASTS" if "seq3" in prefix
                  else "DEV_SEQ2_CONTRASTS"),
        "artifact_prefix": prefix,
        "groups_read": sorted(groups),
        "q1_fixes_and_model": fixes_and_model(groups),
        "q1b_dev_seq1_l2_recomputed": dev_seq1_l2_recomputed(),
        "q2_what_slow_did": what_slow_did(groups),
        "q3_and_q4_per_group": per_group,
        "q5_consistency": consistency(groups, per_group),
        "q6_open_question": open_question(per_group, groups),
        "cost": {"physical_consumer_fits": total_fits,
                 "llm_calls": total_llm,
                 "wall_seconds": round(total_wall, 1),
                 "ledgers_by_run_id": {k: {n: int(v) if n != "wall" else round(v, 1)
                                           for n, v in slot.items()}
                                       for k, slot in sorted(by_run.items())},
                 "aggregation_rule": (
                     "groups sharing a run id share one cumulative budget "
                     "file, so their line total is the maximum, not the sum; "
                     "groups with independent run ids are summed"),
                 "ceilings": {"fits": run.MAX_FITS, "llm": run.MAX_LLM,
                              "wall_seconds": run.MAX_WALL_SECONDS}},
        "boundaries_respected": [
            "no sealed material, no TARGET_HELD_IN unit and no +144 face read",
            "Consumer training and serving geometry, the DSL, the risk lines, "
            "the scoring denominators and the verifier are untouched",
            "the model never edits its own approval rule: adoption is decided "
            "by a deterministic check on readings it did not produce",
            "no raw cross-trajectory Episode reaches Fast in either arm",
            "no new hash, manifest platform or ledger; the prediction cache's "
            "own key is reused",
            "historical receipts are not overwritten and nothing is committed",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    import sys
    argv = list(argv if argv is not None else sys.argv[1:])
    suffix = ""
    prefix = "dev_seq2_slow_update_to_fast"
    for arg in argv:
        if arg.startswith("--suffix="):
            suffix = "_" + arg.split("=", 1)[1].strip("_")
        if arg.startswith("--prefix="):
            prefix = arg.split("=", 1)[1]
    report = build(suffix, prefix)
    out = ART / ("%s__contrasts%s.json" % (prefix, suffix))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drafts._plain(report), ensure_ascii=False,
                              indent=1, default=str), encoding="utf-8")
    print(json.dumps(drafts._plain(report), ensure_ascii=False, indent=1,
                     default=str)[:6000])
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
