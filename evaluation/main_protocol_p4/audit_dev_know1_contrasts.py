"""DEV-KNOW-1 analysis: the three questions the work order asks, in order.

1. **Was knowledge formed?**  Which legal Skill cards the formation segment
   produced, which attempt and which verification each came from, and whether
   each is a new program or a revision of one already held.
2. **Did the knowledge change behaviour?**  Which card the test arms actually
   retrieved, whether the retrieval changed what was proposed, what was probed
   and in what order, and what was deployed -- or whether the card was merely
   loaded and nothing downstream moved.  Read off the recorded trajectory; a
   model's own account of itself is not evidence here.
3. **Was the behaviour change worth anything?**  The two arms compared over the
   same unit set on the full served population: Support and delayed gain, harm,
   coverage, authoritative gate passes, Consumer fits, physical LLM calls, and
   how much feedback each arm needed before its first qualifying deployment.

Everything numeric goes through ``compare_on_a_common_denominator``, so both
sides always name the same units and the three cell states stay apart: a cell
that deployed identity is a real reading of exactly 0.0 -- the policy served raw
and delivered nothing -- while a cell with no reading and no identity
deployment is UNKNOWN, which suppresses the complete verdict instead of being
quietly scored.  The comparison is between two **complete deliveries**, never
between a new card's own cells and something else.

0 LLM calls, 0 Consumer fits.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import run_dev_know1_two_arms as K1

ART = base.ROOT / "artifacts" / "main_protocol"
OUT = ART / "dev_know1_contrasts.json"

FACES = ("support", "delayed")


# ---------------------------------------------------------------------------
# readings
# ---------------------------------------------------------------------------

def _reading(cell: Mapping[str, Any], face: str) -> dict[str, Any]:
    """One cell as the comparator's three states see it."""
    reading = cell.get("%s_reading" % face)
    if reading:
        return {**reading, "status": rev.READ}
    if cell.get("deployed_label") == "identity" or cell.get("identity"):
        return {"status": rev.READ, "aggregate_gain": 0.0,
                "harmed_fraction": 0.0, "max_single_series_harm": 0.0,
                "treated": 0, "served": cell.get("served")}
    return {"status": "NO_READING_AND_NOT_AN_IDENTITY_DEPLOYMENT",
            "why": "the cell produced neither a reading nor a legal identity"}


def _cells(report: Mapping[str, Any], arm: str) -> dict[int, Mapping[str, Any]]:
    return {int(c["position"]): c
            for c in (report.get("test") or {}).get("cells", [])
            if c["arm"] == arm}


def _steps(report: Mapping[str, Any], arm: str) -> dict[int, Mapping[str, Any]]:
    rows = ((report.get("test") or {}).get("revision_steps") or {}).get(arm, [])
    return {int(row["position"]): row for row in rows if row.get("position")}


def _program_of(cell: Mapping[str, Any]) -> str:
    return str(cell.get("deployed_label") or "identity")


def _probe_order(cell: Mapping[str, Any]) -> list[str]:
    """The programs this cell actually tried, in the order it tried them."""
    out = []
    for row in (cell.get("probes") or ()):
        program = ">".join(str(step.get("op"))
                           for step in (row.get("program_steps") or ()))
        out.append("%s:%s" % (row.get("candidate_id") or row.get("kind"),
                              program or "identity"))
    return out


def _proposal_payloads(step: Mapping[str, Any]) -> list[str]:
    out = []
    for proposal in (step.get("proposals") or ()):
        out.append(json.dumps(
            {k: v for k, v in proposal.items()
             if k in ("kind", "workflow_program", "scope_clause",
                      "what_changes")},
            sort_keys=True, ensure_ascii=False, default=str))
    return out


# ---------------------------------------------------------------------------
# question 1 -- was knowledge formed
# ---------------------------------------------------------------------------

def knowledge_formed(report: Mapping[str, Any]) -> dict[str, Any]:
    fork = report.get("fork") or {}
    formation = report.get("formation") or {}
    formed = list(fork.get("formed_in_the_formation_segment") or ())
    entry = list(fork.get("cards_at_the_entry_state") or ())

    provenance: list[dict[str, Any]] = []
    for cell in formation.get("cells", []):
        minted = list(cell.get("skills_minted_this_unit") or ())
        if not minted:
            continue
        event = next((e for e in formation.get("verification_events", [])
                      if int(e.get("verified_at_position") or -1)
                      == int(cell["position"])), None)
        provenance.append({
            "skill_ids": minted,
            "minted_at_position": cell["position"],
            "program": cell.get("deployed_label"),
            "arrived_as": cell.get("deployed_via"),
            "delayed_gate_passed": bool(
                (cell.get("delayed_gate") or {}).get("passes")),
            "delayed_reading": (cell.get("delayed_reading") or {}).get(
                "aggregate_gain"),
            "support_reading": (cell.get("support_reading") or {}).get(
                "aggregate_gain"),
            "proposed_at_position": (event or {}).get("proposed_at_position"),
            "proposed_as_a_change_to": (event or {}).get(
                "parent_program_at_proposal"),
            "what_the_proposal_said_it_changes": (event or {}).get(
                "what_changes"),
            "draft_id": (event or {}).get("draft_id"),
            "draft_closed_as": (event or {}).get("draft_closed_as"),
            "independent_unit": (event or {}).get("independent_unit"),
            "is_a_new_program_or_a_revision": (
                "a revision of a program the library already held"
                if (event or {}).get("parent_program_at_proposal")
                and str(cell.get("deployed_label") or "").split("(")[0]
                != str((event or {}).get("parent_program_at_proposal")
                       or "").split("(")[0]
                and (event or {}).get("proposed_at_position") is not None
                else "a program the library did not hold"),
        })

    attempts = formation.get("attempt_log", [])
    outcomes: dict[str, int] = {}
    for row in attempts:
        outcomes[str(row.get("outcome"))] = outcomes.get(
            str(row.get("outcome")), 0) + 1
    steps = formation.get("revision_steps", [])
    return {
        "treatment": fork.get("treatment"),
        "cards_at_the_entry_state": entry,
        "cards_at_the_boundary": list(fork.get("cards_at_the_boundary") or ()),
        "formed_in_the_formation_segment": formed,
        "lost_during_the_formation_segment": list(
            fork.get("lost_during_the_formation_segment") or ()),
        "how_each_one_arrived": provenance,
        "outer_calls_made": len(steps),
        "outer_call_outcomes": {
            str(step.get("outcome")): sum(
                1 for s in steps if s.get("outcome") == step.get("outcome"))
            for step in steps},
        "proposals_evaluated": len(attempts),
        "proposal_outcomes": outcomes,
        "candidates_queued_but_never_verified": list(
            (formation.get("pending_never_verified") or {})),
        "verification_events": [
            {k: v for k, v in event.items() if k != "delayed_gate"}
            for event in formation.get("verification_events", [])],
        "formation_summary": formation.get("summary"),
        "no_card_was_patched": True,
        "if_nothing_formed": (
            "NO_KNOWLEDGE_TREATMENT: the formation segment produced no usable "
            "new card.  No card is added by hand, and this is not evidence "
            "that knowledge use is ineffective -- it is the absence of a "
            "treatment to test." if not formed else None),
    }


# ---------------------------------------------------------------------------
# question 2 -- did the knowledge change behaviour
# ---------------------------------------------------------------------------

def knowledge_changed_behaviour(report: Mapping[str, Any]) -> dict[str, Any]:
    fork = report.get("fork") or {}
    formed = set(fork.get("formed_in_the_formation_segment") or ())
    old = _cells(report, K1.ARM_OLD)
    acc = _cells(report, K1.ARM_ACC)
    old_steps = _steps(report, K1.ARM_OLD)
    acc_steps = _steps(report, K1.ARM_ACC)
    positions = sorted(set(old) & set(acc))

    per_unit = []
    for position in positions:
        a, b = old[position], acc[position]
        retrieved_formed = [s for s in (b.get("retrieved_skill_ids") or ())
                            if s in formed]
        deployed_formed = b.get("deployed_skill_id") in formed
        row = {
            "position": position,
            "the_formed_card_was_retrieved": retrieved_formed,
            "the_formed_card_was_probed": any(
                str(row.get("candidate_id") or "")[len("cand_skill_"):] in formed
                for row in (b.get("probes") or ())),
            "the_formed_card_was_deployed": bool(deployed_formed),
            "deployed": {K1.ARM_OLD: _program_of(a), K1.ARM_ACC: _program_of(b)},
            "deployment_differs": _program_of(a) != _program_of(b),
            "arrived_as": {K1.ARM_OLD: a.get("deployed_via"),
                           K1.ARM_ACC: b.get("deployed_via")},
            "probe_order": {K1.ARM_OLD: _probe_order(a),
                            K1.ARM_ACC: _probe_order(b)},
            "probe_order_differs": _probe_order(a) != _probe_order(b),
            "candidates_considered": {
                K1.ARM_OLD: sorted(
                    (a.get("candidate_program_steps") or {}).values()),
                K1.ARM_ACC: sorted(
                    (b.get("candidate_program_steps") or {}).values())},
            "candidate_set_differs": sorted(
                (a.get("candidate_program_steps") or {}).values()) != sorted(
                (b.get("candidate_program_steps") or {}).values()),
            "proposals": {
                K1.ARM_OLD: _proposal_payloads(old_steps.get(position, {})),
                K1.ARM_ACC: _proposal_payloads(acc_steps.get(position, {}))},
            "proposals_differ": (
                _proposal_payloads(old_steps.get(position, {}))
                != _proposal_payloads(acc_steps.get(position, {}))),
            "outer_outcome": {
                K1.ARM_OLD: (old_steps.get(position) or {}).get("outcome"),
                K1.ARM_ACC: (acc_steps.get(position) or {}).get("outcome")},
            "cards_held": {
                K1.ARM_OLD: a.get("capability_cards_at_end"),
                K1.ARM_ACC: b.get("capability_cards_at_end")},
        }
        per_unit.append(row)

    retrieved_anywhere = sorted({s for row in per_unit
                                 for s in row["the_formed_card_was_retrieved"]})
    deployed_anywhere = [row["position"] for row in per_unit
                         if row["the_formed_card_was_deployed"]]
    tried_anywhere = [row["position"] for row in per_unit
                      if row["the_formed_card_was_probed"]]
    any_difference = any(row["deployment_differs"] or row["probe_order_differs"]
                         or row["candidate_set_differs"]
                         or row["proposals_differ"] for row in per_unit)
    delivered_differs = [row["position"] for row in per_unit
                         if row["deployment_differs"]]
    # Four places a card can stop, kept apart because each calls for a
    # different response.  "It was loaded" and "it was tried" and "it was
    # deployed" are three different claims, and collapsing them is how a
    # retrieval problem gets mistaken for a content problem.
    if not retrieved_anywhere:
        verdict = "NOT_RETRIEVED"
    elif not tried_anywhere:
        verdict = "RETRIEVED_BUT_NEVER_TRIED"
    elif not deployed_anywhere:
        verdict = "TRIED_BUT_NEVER_SELECTED"
    elif not delivered_differs:
        verdict = "DEPLOYED_BUT_THE_DELIVERY_WAS_THE_SAME"
    else:
        verdict = "DEPLOYED_AND_THE_DELIVERY_DIFFERED"
    return {
        "cards_the_accumulated_arm_could_read": sorted(formed),
        "cards_actually_retrieved_by_the_accumulated_arm": retrieved_anywhere,
        "units_where_a_formed_card_was_probed": tried_anywhere,
        "units_where_a_formed_card_was_deployed": deployed_anywhere,
        "units_where_the_deployment_differed": [
            row["position"] for row in per_unit if row["deployment_differs"]],
        "units_where_the_probe_order_differed": [
            row["position"] for row in per_unit if row["probe_order_differs"]],
        "units_where_the_candidate_set_differed": [
            row["position"] for row in per_unit if row["candidate_set_differs"]],
        "units_where_the_proposals_differed": [
            row["position"] for row in per_unit if row["proposals_differ"]],
        "loaded_but_inert": bool(retrieved_anywhere) and not any_difference,
        "it_changed_what_was_considered_or_tried": any_difference,
        "it_changed_what_was_delivered": bool(delivered_differs),
        "verdict": verdict,
        "read_from": ("the recorded trajectory: retrieved_skill_ids, the probe "
                      "list, the candidate map, the deployed program and the "
                      "outer call's proposals; no model self-report is used"),
        "per_unit": per_unit,
    }


# ---------------------------------------------------------------------------
# question 3 -- was the change worth anything
# ---------------------------------------------------------------------------

def _contrast(report: Mapping[str, Any], face: str) -> dict[str, Any]:
    a = _cells(report, K1.ARM_ACC)
    b = _cells(report, K1.ARM_OLD)
    positions = sorted(set(a) & set(b))
    units = [{"position": p, "unit": a[p]["unit"]} for p in positions]
    out = rev.compare_on_a_common_denominator(
        units=units,
        candidate={"pos%d" % p: rev.classify_reading(_reading(a[p], face))
                   for p in positions},
        reference={"pos%d" % p: rev.classify_reading(_reading(b[p], face))
                   for p in positions},
        face="%s_face" % face,
        candidate_label=K1.ARM_ACC, reference_label=K1.ARM_OLD,
        comparison_kind=("accumulated knowledge against old knowledge on the "
                         "units both ran, full served population, identity "
                         "scored as the 0.0 it actually delivers"))
    gate = {name: sum(1 for p in positions
                      if ((_cells(report, name)[p].get("delayed_gate") or {})
                          .get("passes")))
            for name in (K1.ARM_ACC, K1.ARM_OLD)}
    treated = {name: sum(int(_reading(_cells(report, name)[p], face).get(
                   "treated") or 0) for p in positions)
               for name in (K1.ARM_ACC, K1.ARM_OLD)}
    harmed = {name: sum(int((_cells(report, name)[p].get(
                  "%s_reading" % face) or {}).get("harmed_series") or 0)
                        for p in positions)
              for name in (K1.ARM_ACC, K1.ARM_OLD)}
    out["coverage"] = {"treated_series": treated,
                       "difference": treated[K1.ARM_ACC] - treated[K1.ARM_OLD]}
    out["harm"] = {"harmed_series": harmed,
                   "difference": harmed[K1.ARM_ACC] - harmed[K1.ARM_OLD]}
    out["authoritative_gate_passes"] = {
        **gate, "difference": gate[K1.ARM_ACC] - gate[K1.ARM_OLD]}
    out["units_where_the_deployment_differed"] = [
        p for p in positions
        if a[p].get("deployed_label") != b[p].get("deployed_label")]
    return out


def _cost_and_search(report: Mapping[str, Any]) -> dict[str, Any]:
    """What each arm spent, and how much feedback its first pass cost."""
    out: dict[str, Any] = {}
    cost = report.get("actual_cost") or {}
    group = report.get("group")
    for arm in (K1.ARM_OLD, K1.ARM_ACC):
        cells = _cells(report, arm)
        steps = _steps(report, arm)
        positions = sorted(cells)
        fits = llm = 0
        first: dict[str, Any] | None = None
        for position in positions:
            fits += int(cells[position].get("package_fits_this_cell") or 0)
            llm += int(cells[position].get("package_llm_this_cell") or 0)
            if first is None and bool(
                    (cells[position].get("delayed_gate") or {}).get("passes")):
                first = {
                    "position": position,
                    "units_consumed": positions.index(position) + 1,
                    "consumer_fits_to_here": fits,
                    "physical_llm_calls_to_here": llm,
                    "program": cells[position].get("deployed_label"),
                    "arrived_as": cells[position].get("deployed_via"),
                }
            step = steps.get(position) or {}
            fits += int(step.get("package_fits_this_step") or 0)
            llm += int(step.get("package_llm_this_step") or 0)
        out[arm] = {
            "test_segment_consumer_fits": fits,
            "test_segment_physical_llm_calls": llm,
            "package_ledger_fits": (cost.get(
                "physical_consumer_fits_by_group_and_arm") or {}).get(
                "%s/%s" % (group, arm)),
            "package_ledger_llm": (cost.get(
                "llm_calls_by_group_and_arm") or {}).get(
                "%s/%s" % (group, arm)),
            "feedback_to_the_first_qualifying_deployment": first or {
                "position": None,
                "why": ("no unit in the test segment passed the authoritative "
                        "delayed gate for this arm"),
                "consumer_fits_spent": fits,
                "physical_llm_calls_spent": llm},
        }
    old, acc = out[K1.ARM_OLD], out[K1.ARM_ACC]
    first_old = old["feedback_to_the_first_qualifying_deployment"]
    first_acc = acc["feedback_to_the_first_qualifying_deployment"]
    out["difference"] = {
        "consumer_fits": acc["test_segment_consumer_fits"]
                         - old["test_segment_consumer_fits"],
        "physical_llm_calls": acc["test_segment_physical_llm_calls"]
                              - old["test_segment_physical_llm_calls"],
        "units_to_the_first_qualifying_deployment": (
            (first_acc.get("units_consumed") - first_old.get("units_consumed"))
            if first_acc.get("units_consumed") and first_old.get(
                "units_consumed") else "UNKNOWN"),
        "fits_to_the_first_qualifying_deployment": (
            (first_acc.get("consumer_fits_to_here")
             - first_old.get("consumer_fits_to_here"))
            if first_acc.get("consumer_fits_to_here") is not None
            and first_old.get("consumer_fits_to_here") is not None
            else "UNKNOWN"),
    }
    return out


def was_it_worth_it(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "denominator": ("the test units both arms completed, full served "
                        "population, per-unit pairing, UNKNOWN preserved"),
        "faces": {face: _contrast(report, face) for face in FACES},
        "cost_and_search": _cost_and_search(report),
        "per_arm_delivery": report.get("per_arm"),
        "what_is_compared": ("the complete delivery of each arm over the same "
                             "units, not the new card's own cells"),
    }


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def build(groups: Sequence[str] = ("g1", "g2"),
          suffix: str = "") -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for group in groups:
        path = ART / ("dev_know1_two_arms__%s%s.json" % (group, suffix))
        if path.is_file():
            reports[group] = json.loads(path.read_text(encoding="utf-8"))

    by_group: dict[str, Any] = {}
    for group, report in reports.items():
        by_group[group] = {
            "registered_plan": report.get("registered_plan"),
            "stopped_at": {
                "formation": (report.get("formation") or {}).get("stopped_at"),
                "test": (report.get("test") or {}).get("stopped_at")},
            "test_units_completed_by_both_arms": (
                report.get("geometry") or {}).get(
                "test_units_completed_by_both_arms"),
            "1_was_knowledge_formed": knowledge_formed(report),
            "2_did_it_change_behaviour": knowledge_changed_behaviour(report),
            "3_was_it_worth_anything": was_it_worth_it(report),
            "fork": {k: v for k, v in (report.get("fork") or {}).items()
                     if k != "candidate_supply_pool_closed_at_the_fork"},
            "cost": report.get("actual_cost"),
        }

    treated_groups = [g for g, block in by_group.items()
                      if block["1_was_knowledge_formed"]["treatment"]
                      == "KNOWLEDGE_TREATMENT"]
    untreated_groups = [g for g in by_group if g not in treated_groups]

    headline: dict[str, Any] = {}
    for face in FACES:
        per_group = {}
        for group, block in by_group.items():
            row = block["3_was_it_worth_anything"]["faces"][face]
            per_group[group] = {
                "difference": row["mean_aggregate_gain"]["difference"],
                "accumulated": row["mean_aggregate_gain"]["candidate"],
                "old_knowledge": row["mean_aggregate_gain"]["reference"],
                "coverage_difference": row["coverage"]["difference"],
                "harm_difference": row["harm"]["difference"],
                "gate_difference": row["authoritative_gate_passes"][
                    "difference"],
                "verdict_is_complete": row["verdict_is_complete"],
                "units_where_the_deployment_differed": row[
                    "units_where_the_deployment_differed"],
            }
        signs = {group: (0 if row["difference"] is None
                         else (1 if row["difference"] > 0
                               else -1 if row["difference"] < 0 else 0))
                 for group, row in per_group.items()}
        treated_signs = {g: signs[g] for g in treated_groups if g in signs}
        headline[face] = {
            # Only a group that actually formed a card carries a contrast.  A
            # group with no treatment is two mechanically identical arms, and
            # its difference is the arm-to-arm noise floor -- the scale the
            # treated group's number has to be read against, never a second
            # measurement of the same thing.
            "groups_with_a_treatment": treated_groups,
            "groups_with_no_treatment_read_as_a_noise_floor": untreated_groups,
            "per_group": per_group,
            "signs": signs,
            "treatment_signs": treated_signs,
            "sign_agrees_across_the_treated_groups":
                len(set(treated_signs.values())) == 1 if treated_signs else None,
            "arm_to_arm_noise_floor": {
                group: per_group[group]["difference"]
                for group in untreated_groups if group in per_group},
        }

    treatments = {group: block["1_was_knowledge_formed"]["treatment"]
                  for group, block in by_group.items()}
    behaviour = {group: block["2_did_it_change_behaviour"]["verdict"]
                 for group, block in by_group.items()}
    transports = {group: report.get("transport")
                  for group, report in reports.items()}
    divergences = {group: report.get("transport_divergence")
                   for group, report in reports.items()
                   if report.get("transport_divergence")}
    return {
        "stage": "DEV_KNOW1_CONTRASTS",
        "transport": transports,
        "transport_divergence": divergences or None,
        "comparability": (
            "this run used a transport the previous package did not; its "
            "numbers stand on their own and are not to be differenced against "
            "DEV-AUTO-2/3 or against the deepseek-chat run"
            if divergences else
            "same transport as the previous package"),
        "question": ("under the same subsequent adaptation budget, does the "
                     "Skill accumulated in the formation segment let the agent "
                     "choose, modify and verify programs better than old "
                     "knowledge alone"),
        "groups_read": list(reports),
        "treatment_by_group": treatments,
        "behaviour_verdict_by_group": behaviour,
        "headline": headline,
        "closing_rule": _closing_rule(treatments, behaviour, headline),
        "by_group": by_group,
        "what_this_is_not": [
            "not an independent generalisation proof: five already-exposed "
            "development units, chosen by the course's sequence structure",
            "two groups is a repeat-run check, not independent data",
            "no significance is claimed and no run was added to reach a sign",
        ],
    }


def _closing_rule(treatments: Mapping[str, str], behaviour: Mapping[str, str],
                  headline: Mapping[str, Any]) -> dict[str, Any]:
    """The work order's own closing rule, applied to what was measured."""
    if not treatments:
        return {"rule": "NO_RUN_READ", "then": "nothing to close"}
    if all(value == "NO_KNOWLEDGE_TREATMENT" for value in treatments.values()):
        return {"rule": "NO_KNOWLEDGE_TREATMENT",
                "then": ("no card was formed, so there is nothing to test.  No "
                         "card is added by hand and this is not read as "
                         "'knowledge use is ineffective'.")}
    # Only groups that formed a card say anything about knowledge use.
    treated = {group: value for group, value in behaviour.items()
               if treatments.get(group) == "KNOWLEDGE_TREATMENT"}
    if not treated:
        return {"rule": "NO_KNOWLEDGE_TREATMENT",
                "then": ("no group formed a card, so there is nothing to test")}
    if all(value == "NOT_RETRIEVED" for value in treated.values()):
        return {"rule": "KNOWLEDGE_WAS_NOT_USED",
                "then": ("locate the retrieval entry point; do not add more "
                         "cards")}
    if all(value in ("RETRIEVED_BUT_NEVER_TRIED", "TRIED_BUT_NEVER_SELECTED")
           for value in treated.values()):
        return {"rule": "KNOWLEDGE_WAS_NOT_USED",
                "then": ("retrieval is not the blocker -- the card reached the "
                         "candidate menu, and in the probed case it was even "
                         "measured -- so the use entry point to locate is "
                         "*selection*: what decides which candidate is "
                         "deployed.  Do not add more cards, and do not change "
                         "how they are stored.")}
    if all(value == "DEPLOYED_BUT_THE_DELIVERY_WAS_THE_SAME"
           for value in treated.values()):
        return {"rule": "LOADED_BUT_INERT",
                "then": ("check the card's content and its applicability "
                         "conditions; do not keep changing how it is stored")}
    delayed = headline.get("delayed", {})
    agrees = bool(delayed.get("sign_agrees_across_the_treated_groups"))
    signs = set((delayed.get("treatment_signs") or {}).values())
    if agrees and signs == {1}:
        return {"rule": "BETTER_UNDER_THE_SAME_BUDGET",
                "then": ("proceed to a separate, independent-situation "
                         "validation plan")}
    if agrees and signs == {-1}:
        return {"rule": "NEGATIVE",
                "then": ("report it in full; do not add runs until a positive "
                         "sign appears")}
    if signs == {0}:
        return {"rule": "NO_DIFFERENCE",
                "then": ("the two arms delivered the same thing; report it in "
                         "full and do not add runs to chase a sign")}
    return {"rule": "UNSTABLE",
            "then": ("the treated groups do not agree in sign; report it in "
                     "full and do not add runs until a positive sign appears")}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or [])
    suffix = ""
    for arg in list(argv):
        if arg.startswith("suffix="):
            suffix = "_" + arg.split("=", 1)[1].strip("_")
            argv.remove(arg)
    result = build(tuple(argv) if argv else ("g1", "g2"), suffix=suffix)
    out = ART / ("dev_know1_contrasts%s.json" % suffix)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                              default=str), encoding="utf-8")
    print("groups:", result["groups_read"])
    print("treatment:", result["treatment_by_group"])
    print("behaviour:", result["behaviour_verdict_by_group"])
    for face, block in result["headline"].items():
        print("  %-8s treated-groups agree=%s"
              % (face, block["sign_agrees_across_the_treated_groups"]))
        for group, row in block["per_group"].items():
            print("     %-4s %-13s %+.6f  (acc %s vs old %s) | coverage %+d | "
                  "harm %+d | gate %+d | complete=%s"
                  % (group,
                     "TREATED" if group in block["groups_with_a_treatment"]
                     else "noise-floor",
                     row["difference"] or 0.0, row["accumulated"],
                     row["old_knowledge"], row["coverage_difference"],
                     row["harm_difference"], row["gate_difference"],
                     row["verdict_is_complete"]))
    print("treated groups:", result["headline"]["delayed"][
        "groups_with_a_treatment"], "| noise-floor groups:",
        result["headline"]["delayed"][
            "groups_with_no_treatment_read_as_a_noise_floor"])
    print("closing rule:", result["closing_rule"]["rule"])
    print("  ", result["closing_rule"]["then"])
    print("  wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
