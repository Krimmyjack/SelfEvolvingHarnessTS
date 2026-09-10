"""DEV-AUTO-2 attribution: which channel produced the arm difference.

The arm summary says the revising arm delivered more.  That is not yet an
answer, because the revising arm has **two** channels the control does not:

* **supply** -- a queued candidate becomes a rights-less Draft and is offered
  to later units through ``resupplied_programs_for_verification``.  Fast can
  deploy it before any promotion has happened;
* **the Skill card** -- after a promotion, later Fast retrieves a card whose
  frozen body is the revised program.

DEV-AUTO-2 opens both at once, so the headline difference confounds them.
This reads the course cell by cell and splits it, using the route the runner
already recorded (``deployed_via``) rather than a new inference.  It also
reports the cells where the two arms were bit-identical, which is what makes
the attribution possible at all.

0 LLM calls, 0 Consumer fits: every number here is already in the receipt.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R

ART = base.ROOT / "artifacts" / "main_protocol"
OUT = ART / "dev_auto2_attribution.json"

#: How a deployment reached the unit, and what each route means for the
#: question "did the *Skill* change later behaviour".
ROUTES = {
    "resupplied_draft": "the supply channel: a queued candidate offered to "
                        "this unit; the card may still be the old one",
    "recalled_skill": "the Skill card: Fast retrieved the card and deployed "
                      "its frozen program",
    "searched_active_program": "Fast's own search over programs it already "
                               "holds active",
    "searched_this_unit": "Fast searched and built the program at this unit",
    "identity": "no program deployed",
}


def _gain(cell: Mapping[str, Any], face: str) -> float:
    reading = cell.get("%s_reading" % face) or {}
    return float(reading.get("aggregate_gain") or 0.0)


def build(source: str = "run1") -> dict[str, Any]:
    report = json.loads(
        (ART / ("dev_auto2_frame_c__%s.json" % source)).read_text(
            encoding="utf-8"))
    skill_id = R.K0_SKILL_ID
    by_position: dict[int, dict[str, Any]] = {}
    for cell in report["cells"]:
        by_position.setdefault(int(cell["position"]), {})[cell["arm"]] = cell

    promotions = {int(p["verified_at_position"]): p
                  for p in report["promotions_to_the_skill_card"]}
    rows: list[dict[str, Any]] = []
    for position in sorted(by_position):
        frozen = by_position[position].get(R.FROZEN_ARM)
        revising = by_position[position].get(R.REVISING_ARM)
        if frozen is None or revising is None:
            continue
        same = (frozen.get("deployed_label") == revising.get("deployed_label"))
        row = {
            "position": position,
            "frozen": {"deployed": frozen.get("deployed_label"),
                       "via": frozen.get("deployed_via"),
                       "support_gain": _gain(frozen, "support"),
                       "delayed_gain": _gain(frozen, "delayed"),
                       "gate": (frozen.get("delayed_gate") or {}).get("passes")},
            "revising": {"deployed": revising.get("deployed_label"),
                         "via": revising.get("deployed_via"),
                         "support_gain": _gain(revising, "support"),
                         "delayed_gain": _gain(revising, "delayed"),
                         "gate": (revising.get("delayed_gate")
                                  or {}).get("passes")},
            "identical_deployment": same,
            "delayed_difference": round(_gain(revising, "delayed")
                                        - _gain(frozen, "delayed"), 6),
            "card_body_at_start": {
                "frozen": (frozen.get("skill_bodies_at_start")
                           or {}).get(skill_id),
                "revising": (revising.get("skill_bodies_at_start")
                             or {}).get(skill_id)},
        }
        row["attributed_to"] = (
            "no_difference" if same
            else ROUTES.get(str(revising.get("deployed_via")),
                            str(revising.get("deployed_via"))))
        row["channel"] = (
            "none" if same else
            "supply" if revising.get("deployed_via") == "resupplied_draft" else
            "skill_card" if revising.get("deployed_via") == "recalled_skill"
            else "fast_search_under_the_new_card")
        rows.append(row)

    divergent = [r for r in rows if not r["identical_deployment"]]
    by_channel: dict[str, dict[str, Any]] = {}
    for row in divergent:
        entry = by_channel.setdefault(row["channel"],
                                      {"units": [], "delayed_difference": 0.0})
        entry["units"].append(row["position"])
        entry["delayed_difference"] = round(
            entry["delayed_difference"] + row["delayed_difference"], 6)

    frozen_cards = {r["card_body_at_start"]["frozen"] for r in rows}
    revising_cards = [r["card_body_at_start"]["revising"] for r in rows]
    card_changes = [rows[i]["position"] for i in range(1, len(rows))
                    if revising_cards[i] != revising_cards[i - 1]]

    used_after = []
    for row in rows:
        if row["channel"] == "skill_card":
            used_after.append(row["position"])

    return {
        "stage": "DEV_AUTO2_ATTRIBUTION",
        "what_this_is": ("which of the revising arm's two channels produced "
                         "the difference, read off the route the runner "
                         "recorded; no new inference and no new reading"),
        "source": "artifacts/main_protocol/dev_auto2_frame_c__%s.json" % source,
        "units_compared": len(rows),
        "units_with_an_identical_deployment": sum(
            1 for r in rows if r["identical_deployment"]),
        "units_that_diverged": [r["position"] for r in divergent],
        "why_that_matters": (
            "the arms took the same action on %d of %d units, so the whole "
            "difference sits in %d cells and can be attributed cell by cell "
            "rather than assumed"
            % (sum(1 for r in rows if r["identical_deployment"]), len(rows),
               len(divergent))),
        "difference_by_channel": by_channel,
        "total_delayed_difference": round(
            sum(r["delayed_difference"] for r in rows), 6),
        "the_skill_card": {
            "frozen_arm_distinct_bodies": len(frozen_cards),
            "the_control_never_revised_its_card": len(frozen_cards) == 1,
            "revising_arm_card_changed_at_positions": card_changes,
            "promotions": [
                {"program": p["program"],
                 "proposed_at_position": p["proposed_at_position"],
                 "verified_at_position": p["verified_at_position"],
                 "surface": p["surface"], "cause": p["cause"],
                 "compiled_and_materialized": p.get("compiled_and_materialized"),
                 "body_readback": p.get("body_readback"),
                 "draft_closed_as": p.get("draft_closed_as")}
                for p in report["promotions_to_the_skill_card"]],
            "units_where_a_promoted_card_was_retrieved_and_deployed": used_after,
            "note": ("a promotion is not use.  The second promoted card was "
                     "never deployed: at the units after it Fast searched and "
                     "chose the plain parent program instead"),
        },
        "funnel": _funnel(report),
        "per_unit": rows,
        "cost": report["actual_cost"],
    }


def _funnel(report: Mapping[str, Any]) -> dict[str, Any]:
    steps = report["revision_steps"]
    proposals = [e for s in steps for e in (s.get("evaluations") or ())]
    queued = [e for e in proposals
              if e.get("frame_c_outcome") == "QUEUED_FOR_VERIFICATION"]
    opened = [e for e in queued if (e.get("written_back") or {}).get("applied")]
    would_have = {}
    for e in queued:
        name = str((e.get("the_absolute_screen_would_have_said")
                    or {}).get("outcome"))
        would_have[name] = would_have.get(name, 0) + 1
    ledger = report["draft_ledgers"][R.REVISING_ARM]
    return {
        "revision_calls": len(steps),
        "call_outcomes": _count(steps, "outcome"),
        "proposals": len(proposals),
        "by_kind": _count(proposals, "kind"),
        "frame_c_outcomes": _count(proposals, "frame_c_outcome"),
        "queued": len(queued),
        "drafts_opened": len(opened),
        "queued_but_the_lineage_already_had_a_draft": len(queued) - len(opened),
        "what_the_absolute_screen_would_have_done_to_the_queued": would_have,
        "under_frame_a_or_b_none_of_these_would_have_entered": all(
            name in ("RISK_LINE_FAILED", "NOT_READABLE")
            for name in would_have),
        "drafts_by_closure": _count(ledger["drafts"], "closed"),
        "promoted": len(report["promotions_to_the_skill_card"]),
        "lifecycle_bounds": {
            "max_revisions": drafts.MAX_REVISIONS,
            "max_verification_attempts": drafts.MAX_VERIFICATION_ATTEMPTS},
    }


def _count(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[str(row.get(key))] = out.get(str(row.get(key)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or [])
    result = build(argv[0] if argv else "run1")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                              default=str), encoding="utf-8")
    print("units compared %s | identical %s | diverged %s"
          % (result["units_compared"],
             result["units_with_an_identical_deployment"],
             result["units_that_diverged"]))
    for channel, entry in result["difference_by_channel"].items():
        print("  %-32s units %-14s delayed %+.6f"
              % (channel, entry["units"], entry["delayed_difference"]))
    print("  total delayed difference %+.6f" % result["total_delayed_difference"])
    print("  funnel:", json.dumps(result["funnel"], ensure_ascii=False)[:400])
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
