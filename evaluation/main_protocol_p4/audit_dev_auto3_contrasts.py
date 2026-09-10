"""DEV-AUTO-3 analysis: B-A, C-B, C-A, on one denominator, over both groups.

Three differences and what each one is:

* **B - A** -- what proposing and supplying new candidates is worth.
* **C - B** -- what additionally writing a verified candidate onto the target
  Skill card is worth.
* **C - A** -- the whole revision channel.

Every comparison goes through ``compare_on_a_common_denominator``, so the two
sides always name the same unit set and the three cell states stay apart.  A
cell where the arm deployed identity is a real reading of exactly 0.0 -- the
policy served raw and delivered nothing, which is not the same as a cell whose
reading is missing.  A cell with no reading and no identity deployment is
UNKNOWN and suppresses the complete verdict.

The behaviour route (``recalled_skill`` and friends) is carried through for
explanation only.  Gain that arrived through ``recalled_skill`` is **not** the
causal value of the Skill update: in this design an arm that never PATCHes the
target card can still mint and recall an ordinary Skill through the normal Fast
lifecycle, which is exactly what arm B is allowed to keep.

0 LLM calls, 0 Consumer fits.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import run_dev_auto3_three_arms as D3

ART = base.ROOT / "artifacts" / "main_protocol"
OUT = ART / "dev_auto3_contrasts.json"

FACES = ("support", "delayed")
CONTRASTS = (("B_minus_A", D3.ARM_B, D3.ARM_A,
              "what proposing and supplying candidates is worth"),
             ("C_minus_B", D3.ARM_C, D3.ARM_B,
              "what additionally writing the card is worth"),
             ("C_minus_A", D3.ARM_C, D3.ARM_A,
              "the whole revision channel"))


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
    return {int(c["position"]): c for c in report["cells"] if c["arm"] == arm}


def _contrast(report: Mapping[str, Any], mine: str, theirs: str,
              face: str) -> dict[str, Any]:
    a, b = _cells(report, mine), _cells(report, theirs)
    positions = sorted(set(a) & set(b))
    units = [{"position": p, "unit": a[p]["unit"]} for p in positions]
    out = rev.compare_on_a_common_denominator(
        units=units,
        candidate={"pos%d" % p: rev.classify_reading(_reading(a[p], face))
                   for p in positions},
        reference={"pos%d" % p: rev.classify_reading(_reading(b[p], face))
                   for p in positions},
        face="%s_face" % face, candidate_label=mine, reference_label=theirs,
        comparison_kind=("arm against arm on the units both ran, full served "
                         "population, identity scored as the 0.0 it delivers"))
    gate = {name: sum(1 for p in positions
                      if ((_cells(report, name)[p].get("delayed_gate") or {})
                          .get("passes")))
            for name in (mine, theirs)}
    treated = {name: sum(int((_reading(_cells(report, name)[p], face)
                              .get("treated") or 0)) for p in positions)
               for name in (mine, theirs)}
    harmed = {name: sum(int((_cells(report, name)[p].get(
                  "%s_reading" % face) or {}).get("harmed_series") or 0)
                        for p in positions)
              for name in (mine, theirs)}
    out["coverage"] = {"treated_series": treated,
                       "difference": treated[mine] - treated[theirs]}
    out["harm"] = {"harmed_series": harmed,
                   "difference": harmed[mine] - harmed[theirs]}
    out["authoritative_gate_passes"] = {**gate,
                                        "difference": gate[mine] - gate[theirs]}
    out["units_where_the_deployment_differed"] = [
        p for p in positions
        if a[p].get("deployed_label") != b[p].get("deployed_label")]
    return out


def _timeline(report: Mapping[str, Any]) -> dict[str, Any]:
    """When each candidate formed, and when knowledge changed."""
    rows: list[dict[str, Any]] = []
    for arm, events in (report.get("verification_events") or {}).items():
        for event in events:
            rows.append({
                "arm": arm,
                "program": event.get("program"),
                "proposed_at_position": event.get("proposed_at_position"),
                "verified_at_position": event.get("verified_at_position"),
                "written_back": bool((event.get("write_back") or {}).get(
                    "applied")),
                "written_back_at_position": (
                    event.get("verified_at_position")
                    if (event.get("write_back") or {}).get("applied") else None),
                "first_later_use": event.get("first_later_use"),
                "parent_program_at_proposal":
                    event.get("parent_program_at_proposal"),
                "card_at_proposal_was_the_ancestor": (
                    event.get("card_body_at_proposal")
                    == (report.get("geometry") or {}).get(
                        "ancestor_card_body")),
                "draft_closed_as": event.get("draft_closed_as"),
            })
    steps = report.get("revision_steps") or {}
    read_an_update = []
    for arm, arm_steps in steps.items():
        for step in arm_steps:
            knowledge = step.get("knowledge_state_at_this_call") or {}
            if knowledge.get("this_call_reads_an_updated_card"):
                read_an_update.append({
                    "arm": arm, "position": step.get("position"),
                    "parent_program_now": knowledge.get("parent_program_now"),
                    "updates_already_landed": knowledge.get(
                        "updates_that_had_already_landed"),
                    "proposals": [
                        (p.get("workflow_program") or p.get("scope_clause"))
                        for p in (step.get("proposals") or ())],
                })
    overwritten = []
    for arm, events in (report.get("verification_events") or {}).items():
        written = [e for e in events
                   if (e.get("write_back") or {}).get("applied")]
        for earlier, later in zip(written, written[1:]):
            if earlier.get("first_later_use") is None:
                overwritten.append({
                    "arm": arm,
                    "overwritten_program": earlier.get("program"),
                    "written_back_at": earlier.get("verified_at_position"),
                    "overwritten_at": later.get("verified_at_position"),
                    "was_ever_used": False})
    return {
        "events": rows,
        "calls_that_read_an_already_updated_card": read_an_update,
        "a_later_proposal_built_on_an_earlier_update": bool(read_an_update),
        "updates_overwritten_before_they_were_ever_used": overwritten,
    }


def _per_arm(report: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for name, summary in (report.get("per_arm") or {}).items():
        cost = report["actual_cost"]
        out[name] = {
            "support_face": summary["support_face"],
            "delayed_face": summary["delayed_face"],
            "deployed_program": summary["deployed_program"],
            "deployed_via": summary["deployed_via"],
            "fits": (cost.get("physical_consumer_fits_by_arm") or {}).get(name),
            "llm_calls": (cost.get("llm_calls_by_arm") or {}).get(name),
        }
    return out


def build(groups: Sequence[str] = ("g1", "g2")) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for group in groups:
        path = ART / ("dev_auto3_three_arms__%s.json" % group)
        if path.is_file():
            reports[group] = json.loads(path.read_text(encoding="utf-8"))

    by_group: dict[str, Any] = {}
    for group, report in reports.items():
        contrasts = {}
        for name, mine, theirs, meaning in CONTRASTS:
            contrasts[name] = {
                "meaning": meaning,
                **{face: _contrast(report, mine, theirs, face)
                   for face in FACES},
            }
        by_group[group] = {
            "registered_plan": report.get("registered_plan"),
            "units_completed_by_every_arm": (report.get("geometry")
                                             or {}).get(
                "units_completed_by_every_arm"),
            "stopped_at": report.get("stopped_at"),
            "per_arm": _per_arm(report),
            "contrasts": contrasts,
            "timeline": _timeline(report),
            "cost": report["actual_cost"],
        }

    headline = {}
    for name, _mine, _theirs, meaning in CONTRASTS:
        per_group = {}
        for group, block in by_group.items():
            face = block["contrasts"][name]["delayed"]
            per_group[group] = {
                "delayed_difference": face["mean_aggregate_gain"]["difference"],
                "coverage_difference": face["coverage"]["difference"],
                "harm_difference": face["harm"]["difference"],
                "gate_difference": face["authoritative_gate_passes"][
                    "difference"],
                "verdict_is_complete": face["verdict_is_complete"],
            }
        signs = {group: (0 if row["delayed_difference"] is None
                         else (1 if row["delayed_difference"] > 0
                               else -1 if row["delayed_difference"] < 0 else 0))
                 for group, row in per_group.items()}
        headline[name] = {
            "meaning": meaning,
            "per_group": per_group,
            "sign_agrees_across_groups": len(set(signs.values())) == 1,
            "signs": signs,
        }

    return {
        "stage": "DEV_AUTO3_CONTRASTS",
        "what_this_is": ("the three differences on one denominator, per group, "
                         "with per-unit pairing and UNKNOWN preserved"),
        "groups_read": list(reports),
        "headline_delayed_face": headline,
        "by_group": by_group,
        "how_to_read_the_routes": (
            "the deployment route is explanation, not attribution: an arm that "
            "never PATCHes the target card can still mint and recall an "
            "ordinary Skill through the normal Fast lifecycle, so gain "
            "arriving via recalled_skill is not by itself the causal value of "
            "the Skill update"),
        "cost": {group: block["cost"] for group, block in by_group.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or [])
    result = build(tuple(argv) if argv else ("g1", "g2"))
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                              default=str), encoding="utf-8")
    print("groups:", result["groups_read"])
    for name, block in result["headline_delayed_face"].items():
        print("  %-12s %-52s agrees=%s" % (name, block["meaning"],
                                           block["sign_agrees_across_groups"]))
        for group, row in block["per_group"].items():
            print("      %-4s delayed %+.6f | coverage %+d | harm %+d | gate %+d | complete=%s"
                  % (group, row["delayed_difference"] or 0.0,
                     row["coverage_difference"], row["harm_difference"],
                     row["gate_difference"], row["verdict_is_complete"]))
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
