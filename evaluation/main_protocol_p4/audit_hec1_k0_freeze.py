"""The K0 freeze gate: five mechanical checks before K0 becomes an arm's start.

Why this is separate from ``audit_hec1_k0``
-------------------------------------------
That one answers "did these cards pass their gates".  This is the **freeze**
gate sol required before Phase T may start, and it asks the questions that only
matter at the moment K0 stops being a result and starts being an input:

* does the receipt describe the course it claims to come from, or has it drifted
  from it;
* does a non-empty K0 actually resolve to a materialised snapshot, or is it a
  list of names with nothing behind them;
* is the arm set that follows from K0's emptiness the one the contract declares,
  and is criterion 3 scored only when there is something to score it on;
* can every card prove its route -- Support, the authoritative delayed gate, a
  threshold from the tool rather than from the model, no replay-granted
  activation, and no more clauses than the lifecycle allows;
* is K0 unreachable from the arms that are supposed to start without it.

The last one is the load-bearing one.  K0 is what makes A5 *A5*; a card
reachable from A3 would make every A5−A3 number in the report an artefact of a
leak rather than a measurement of accumulation.

What it may not read
--------------------
Any gain on the evaluation face (checklist H4).  An instrument gate that could
see whether the course looked good is not an instrument gate.  It reads the
delayed gate's pass/fail, the Support probe records, the Scope predicates, the
receipt and the arms' starting Skill sets -- and nothing about utility.

An empty K0 passes.  It is a legal freeze, recorded ``A5_TREATMENT_EMPTY``, the
arm set contracts, and criterion 3 goes unscored -- there is simply nothing whose
route needs proving.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import scope_narrowing_preflight as narrowing
from evaluation.main_protocol_p4 import scope_threshold_tool as tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts/main_protocol"
OUT_JSON = ARTIFACTS / "hec1_k0_freeze.json"

CHECK_NAMES = (
    "receipt_matches_course",
    "snapshot_resolves",
    "arm_set_for_phase_t",
    "card_provenance",
    "a5_a3_isolation",
)

#: What every Active card must be able to show, inside ``card_provenance``.
CARD_ASSERTIONS = (
    "support_reading_exists",
    "delayed_passed_the_authoritative_gate",
    "threshold_came_from_the_tool",
    "no_replay_granted_activation",
    "within_the_clause_budget",
)

#: Arms that may start from K0.  Anything else reaching it is a leak.
K0_ARMS = ("A5-frozen", "A5-online")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _activating_cells(course: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [cell for cell in (course.get("cells") or ()) if cell.get("activated")]


def _course_active_ids(course: Mapping[str, Any]) -> set[str]:
    """Every Skill id the course itself says it activated."""
    seen: set[str] = set()
    for cell in _activating_cells(course):
        seen.update(str(value) for value in (cell.get("active_skill_ids") or ()))
    for ids in (course.get("active_skill_ids") or {}).values():
        seen.update(str(value) for value in (ids or ()))
    return seen


def _clauses(scope: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return [dict(clause) for clause in ((scope or {}).get("predicate") or ())]


def _threshold_provenance(scope: Mapping[str, Any] | None) -> dict[str, Any]:
    """Every clause threshold must sit on a frozen bin edge of its feature."""
    clauses = _clauses(scope)
    if not clauses:
        return {"passed": True, "clauses": 0,
                "why": "no predicate; the card treats every served series"}
    offenders = []
    for clause in clauses:
        feature = str(clause.get("feature"))
        try:
            edges = tool.frozen_bin_edges(feature)
        except Exception as exc:  # noqa: BLE001 - an unknown feature is a fail
            offenders.append({"clause": clause,
                              "why": "unknown feature: %s" % str(exc)[:120]})
            continue
        if float(clause.get("threshold")) not in {float(e) for e in edges}:
            offenders.append({
                "clause": clause, "bin_edges": list(edges),
                "why": ("threshold is not a frozen bin edge, so it did not "
                        "come from the calibration tool")})
    return {"passed": not offenders, "clauses": len(clauses),
            "offenders": offenders}


def _clause_budget(scope: Mapping[str, Any] | None,
                   root: Mapping[str, Any] | None) -> dict[str, Any]:
    """At most two clauses added since the initialiser wrote the root."""
    now = _clauses(scope)
    if root is None:
        within = len(now) <= narrowing.MAX_TOTAL_ADDED_CLAUSES + 1
        return {"passed": within, "clauses": len(now), "added": None,
                "why": ("no root recorded; judged on the absolute clause count, "
                        "which is the weaker check")}
    root_set = {json.dumps(clause, sort_keys=True) for clause in _clauses(root)}
    added = [clause for clause in now
             if json.dumps(clause, sort_keys=True) not in root_set]
    return {"passed": len(added) <= narrowing.MAX_TOTAL_ADDED_CLAUSES,
            "clauses": len(now), "added": len(added),
            "max_total_added": narrowing.MAX_TOTAL_ADDED_CLAUSES}


def audit_card(skill_id: str, evidence: Sequence[Mapping[str, Any]]
               ) -> dict[str, Any]:
    """The five assertions for one Active card, from the cells that made it."""
    support = bool(evidence) and all(cell.get("probes") for cell in evidence)
    gate = bool(evidence) and all(
        ((cell.get("delayed") or {}).get("gate") or {}).get("passes") is True
        and (cell.get("gate_disagreement") or {}).get("resolved_by") == "p4_gate"
        and (cell.get("gate_disagreement") or {}).get("may_activate") is True
        for cell in evidence)
    no_replay = all(str(cell.get("deployed_via")) != "replay"
                    for cell in evidence)
    provenance: dict[str, Any] = {"passed": True, "clauses": 0}
    budget: dict[str, Any] = {"passed": True, "clauses": 0}
    for cell in evidence:
        scope = cell.get("deployed_serving_scope")
        provenance = _threshold_provenance(scope)
        budget = _clause_budget(scope, cell.get("deployed_root_scope"))
        if not provenance["passed"] or not budget["passed"]:
            break
    card = {
        "skill_id": skill_id,
        "activations": len(evidence),
        "support_reading_exists": support,
        "delayed_passed_the_authoritative_gate": gate,
        "threshold_came_from_the_tool": provenance["passed"],
        "threshold_detail": provenance,
        "no_replay_granted_activation": no_replay,
        "within_the_clause_budget": budget["passed"],
        "clause_budget_detail": budget,
        "units": [cell.get("unit") for cell in evidence],
    }
    card["passed"] = all(card[name] for name in CARD_ASSERTIONS)
    return card


# ---------------------------------------------------------------------------
# the five checks
# ---------------------------------------------------------------------------

def _check_receipt_matches_course(course: Mapping[str, Any],
                                  receipt: Mapping[str, Any]) -> dict[str, Any]:
    claimed = {str(value) for value in (receipt.get("active_skill_ids") or ())}
    actual = _course_active_ids(course)
    unbacked = sorted(claimed - actual)
    unlisted = sorted(actual - claimed)
    empty_agrees = bool(receipt.get("empty", not claimed)) == (not claimed)
    return {
        "check": "receipt_matches_course",
        "passed": not unbacked and not unlisted and empty_agrees,
        "claimed": sorted(claimed),
        "activated_in_course": sorted(actual),
        "claimed_but_not_activated": unbacked,
        "activated_but_not_claimed": unlisted,
        "empty_flag_agrees": empty_agrees,
        "why": (
            "a receipt is the handover; if it names a card the course never "
            "activated, Phase T would start from something that was never gated"
        ),
    }


def _check_snapshot_resolves(receipt: Mapping[str, Any]) -> dict[str, Any]:
    empty = bool(receipt.get("empty", not (receipt.get("active_skill_ids") or ())))
    if empty:
        return {"check": "snapshot_resolves", "passed": True, "k0_empty": True,
                "why": "an empty K0 has no snapshot to resolve"}
    resolved = bool(receipt.get("snapshot_resolved"))
    root = receipt.get("store_root")
    sha = receipt.get("runtime_bundle_sha")
    return {
        "check": "snapshot_resolves",
        "passed": resolved and bool(root) and bool(sha),
        "k0_empty": False,
        "snapshot_resolved": resolved,
        "store_root": root,
        "runtime_bundle_sha": sha,
        "why": (
            "a non-empty K0 that does not resolve to a materialised snapshot is "
            "a list of names; the A5 arms would silently start from h0 and the "
            "accumulation contrast would compare two identical arms"
        ),
    }


def _check_arm_set(receipt: Mapping[str, Any]) -> dict[str, Any]:
    empty = bool(receipt.get("empty", not (receipt.get("active_skill_ids") or ())))
    arms = list(contract.ARMS["empty_k0" if empty else "full_k0"])
    return {
        "check": "arm_set_for_phase_t",
        "passed": True,
        "k0_empty": empty,
        "arms": arms,
        "criterion_3_scored": not empty,
        "why": (
            "with an empty K0 the A5 arms would be bit-identical to A3, so they "
            "are not run and the accumulation criterion is not scored; paying "
            "for an equivalent arm would buy no contrast"
        ),
    }


def _check_card_provenance(course: Mapping[str, Any],
                           receipt: Mapping[str, Any]) -> dict[str, Any]:
    active = sorted({str(v) for v in (receipt.get("active_skill_ids") or ())})
    if not active:
        return {"check": "card_provenance", "passed": True, "cards": [],
                "cards_total": 0, "cards_passed": 0,
                "why": "an empty K0 has no card whose route needs proving"}
    activations = _activating_cells(course)
    cards = [
        audit_card(skill_id,
                   [cell for cell in activations
                    if skill_id in (cell.get("active_skill_ids") or ())])
        for skill_id in active
    ]
    failed = [card for card in cards if not card["passed"]]
    return {
        "check": "card_provenance",
        "passed": not failed,
        "assertions": list(CARD_ASSERTIONS),
        "cards": cards,
        "cards_total": len(cards),
        "cards_passed": len(cards) - len(failed),
    }


def _check_isolation(receipt: Mapping[str, Any],
                     target_course: Mapping[str, Any] | None) -> dict[str, Any]:
    cards = {str(v) for v in (receipt.get("active_skill_ids") or ())}
    declared = {name: spec["start"]
                for name, spec in contract.ARMS["full_k0"].items()}
    wrong = sorted(name for name, start in declared.items()
                   if start == "k0" and name not in K0_ARMS)
    leaks: list[dict[str, Any]] = []
    if target_course and cards:
        for cell in target_course.get("cells") or ():
            arm = str(cell.get("arm"))
            if arm in K0_ARMS:
                continue
            seen = {str(v) for v in (cell.get("retrieved_skill_ids") or ())}
            seen |= {str(v) for v in (cell.get("active_skill_ids") or ())}
            overlap = sorted(seen & cards)
            if overlap:
                leaks.append({"arm": arm, "position": cell.get("position"),
                              "k0_cards_seen": overlap})
    return {
        "check": "a5_a3_isolation",
        "passed": not wrong and not leaks,
        "k0_card_count": len(cards),
        "arms_declared_to_start_at_k0": sorted(
            name for name, start in declared.items() if start == "k0"),
        "unexpected_k0_arms": wrong,
        "leaks_into_non_k0_arms": leaks,
        "checked_against_target_course": bool(target_course),
        "why": (
            "K0 is what makes A5 A5; a card reachable from A3 would make every "
            "A5 minus A3 number an artefact of a leak"
        ),
    }


def audit(course: Mapping[str, Any], receipt: Mapping[str, Any],
          target_course: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run all five.  Reads gates, probes, predicates -- never a gain."""
    checks = [
        _check_receipt_matches_course(course, receipt),
        _check_snapshot_resolves(receipt),
        _check_arm_set(receipt),
        _check_card_provenance(course, receipt),
        _check_isolation(receipt, target_course),
    ]
    empty = bool(receipt.get("empty",
                             not (receipt.get("active_skill_ids") or ())))
    passed = all(row["passed"] for row in checks)
    return {
        "stage": "HEC1_K0_FREEZE",
        "written_at": datetime.now().astimezone().isoformat(),
        "contract_version": contract.VERSION,
        "k0_empty": empty,
        "checks": checks,
        "checks_total": len(checks),
        "checks_passed": sum(1 for row in checks if row["passed"]),
        "check_names": list(CHECK_NAMES),
        "passed": passed,
        "may_continue": passed,
        "verdict": ("A5_TREATMENT_EMPTY" if empty and passed
                    else "K0_FREEZE_CLEAN" if passed else "K0_FREEZE_FAILED"),
        "reading": (
            "an empty K0 is a legal freeze: the arm set contracts to Static / "
            "A3-frozen / A3-online, criterion 3 is not scored, and nothing is "
            "re-run to manufacture a treatment" if empty else
            "every card proved its route and K0 is unreachable from A3"
            if passed else "a card could not prove its route, or K0 leaks"),
        "on_failure": (
            "stop; a card that cannot prove its route, or a K0 that leaks into "
            "A3, may not become an arm's starting state"
        ),
        "boundary": {"llm_calls": 0, "consumer_fits": 0, "held_out_reads": 0,
                     "evaluation_gains_read": 0},
    }


def _md(payload: Mapping[str, Any]) -> str:
    lines = [
        "# HEC-1 K0 freeze gate",
        "",
        "Five mechanical checks. Reads gates, probes and predicates; reads "
        "**no** evaluation-face gain.",
        "",
        "| check | state | note |",
        "| --- | --- | --- |",
    ]
    for row in payload["checks"]:
        note = ""
        if row["check"] == "arm_set_for_phase_t":
            note = "%s; criterion 3 scored: %s" % (
                ", ".join(row["arms"]), row["criterion_3_scored"])
        elif row["check"] == "card_provenance":
            note = "%s/%s cards" % (row.get("cards_passed"),
                                   row.get("cards_total"))
        elif not row["passed"]:
            note = row.get("why", "")[:120]
        lines.append("| `%s` | %s | %s |" % (
            row["check"], "PASS" if row["passed"] else "**FAIL**", note))
    lines += ["", "**%s** (%d/%d checks). K0 empty: %s." % (
        payload["verdict"], payload["checks_passed"], payload["checks_total"],
        payload["k0_empty"]), "", payload["reading"] + ".", ""]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("course", help="the Phase S course artifact")
    parser.add_argument("--k0", default="artifacts/main_protocol/hec1_k0.json")
    parser.add_argument("--target-course", default=None,
                       help="a Phase T course, to check A5/A3 isolation live")
    parser.add_argument("--label", default=None)
    args = parser.parse_args(argv)
    started = time.time()
    course = json.loads(Path(args.course).read_text(encoding="utf-8"))
    k0_path = PROJECT_ROOT / args.k0
    receipt = (json.loads(k0_path.read_text(encoding="utf-8"))
               if k0_path.is_file() else {"active_skill_ids": [], "empty": True})
    target = (json.loads(Path(args.target_course).read_text(encoding="utf-8"))
              if args.target_course else None)
    payload = {**audit(course, receipt, target),
               "wall_seconds": round(time.time() - started, 2)}
    out_json = (OUT_JSON if not args.label
                else OUT_JSON.with_name("hec1_k0_freeze_%s.json" % args.label))
    if out_json.exists():
        print("refusing to overwrite %s; pass --label <new label>"
              % out_json.relative_to(PROJECT_ROOT).as_posix())
        return 2
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    out_json.with_suffix(".md").write_text(_md(payload), encoding="utf-8")
    for row in payload["checks"]:
        print("  %-26s %s" % (row["check"],
                             "PASS" if row["passed"] else "FAIL"))
    print("verdict      : %s" % payload["verdict"])
    print("may continue : %s" % payload["may_continue"])
    print("wrote %s" % out_json.relative_to(PROJECT_ROOT).as_posix())
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
