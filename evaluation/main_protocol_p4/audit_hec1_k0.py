"""Can every card in K0 prove how it got there?  Three assertions per card.

Why a card needs a provenance audit at all
-----------------------------------------
K0 is what A5 *is*.  If a card reached it by any route other than the one the
protocol declares, then the A5/A3 contrast stops measuring inherited knowledge
and starts measuring a bookkeeping accident -- and the whole accumulation claim
rests on that contrast.  Three routes have gone wrong on this line before, so
each is now an assertion:

1. **the authoritative gate** -- the card must carry a Support reading and a
   delayed reading that the P4 gate (coverage floor included) passed.  Source-v3
   round 2856 had a lifecycle event saying ``approved`` while the gate said no;
   an ``online_loop`` approval alone is not a provenance.
2. **the threshold is the tool's** -- any Scope clause on the card must sit on a
   frozen bin edge.  A number the model chose would mean the Scope was eyeballed,
   which is the failure the whole W2 chain exists to remove.
3. **no replay granted it** -- the replay screen may eliminate and may not
   promote.  A card whose only evidence is a replay on already-processed cells
   has been verified on the window it was selected from.

An empty K0 is a legal freeze and passes trivially: it is recorded as
``A5_TREATMENT_EMPTY``, the arm set contracts, criterion 3 goes unscored, and
nothing is re-run to manufacture a treatment.

Any failing assertion stops the chain.  0 LLM, 0 fits: the audit reads the course
artifact and the frozen bin edges, nothing else.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import scope_threshold_tool as tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts/main_protocol"
OUT_JSON = ARTIFACTS / "hec1_k0_audit.json"
OUT_MD = ARTIFACTS / "hec1_k0_audit.md"

ASSERTIONS = ("passed_the_authoritative_gate", "threshold_came_from_the_tool",
              "no_replay_granted_activation")


def _activations(course: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every cell that activated something, with the evidence it activated on."""
    rows = []
    for cell in course.get("cells") or ():
        if not cell.get("activated"):
            continue
        rows.append({
            "arm": cell.get("arm"),
            "position": cell.get("position"),
            "unit": cell.get("unit"),
            "program": cell.get("deployed"),
            "serving_scope": cell.get("deployed_serving_scope"),
            "probes": [row for row in (cell.get("probes") or ())
                       if row.get("kind") == "probe"],
            "delayed": cell.get("delayed"),
            "gate_disagreement": cell.get("gate_disagreement"),
            "active_skill_ids": cell.get("active_skill_ids") or [],
            "deployed_via": cell.get("deployed_via"),
        })
    return rows


def _clause_thresholds_are_frozen_edges(scope: Mapping[str, Any] | None
                                        ) -> dict[str, Any]:
    predicate = list((scope or {}).get("predicate") or ())
    if not predicate:
        return {"passed": True, "clauses": 0,
                "why": "no predicate to calibrate; the card treats every "
                       "served series"}
    offenders = []
    for clause in predicate:
        feature = str(clause.get("feature"))
        try:
            edges = tool.frozen_bin_edges(feature)
        except Exception as exc:  # noqa: BLE001 - an unknown feature is a failure
            offenders.append({"clause": dict(clause),
                              "why": "unknown feature: %s" % str(exc)[:120]})
            continue
        if float(clause.get("threshold")) not in {float(e) for e in edges}:
            offenders.append({"clause": dict(clause), "bin_edges": list(edges),
                              "why": "threshold is not a frozen bin edge, so it "
                                     "was not the tool's"})
    return {"passed": not offenders, "clauses": len(predicate),
            "offenders": offenders}


def audit(course: Mapping[str, Any], k0: Mapping[str, Any]) -> dict[str, Any]:
    active = sorted(k0.get("active_skill_ids") or ())
    if not active:
        return {
            "stage": "HEC1_K0_AUDIT",
            "written_at": datetime.now().astimezone().isoformat(),
            "contract_version": contract.VERSION,
            "k0_empty": True,
            "cards": [],
            "passed": True,
            "may_continue": True,
            "verdict": "A5_TREATMENT_EMPTY",
            "reading": (
                "an empty K0 is a legal freeze: the arm set contracts to Static "
                "/ A3-frozen / A3-online, criterion 3 is not scored, and nothing "
                "is re-run to manufacture a treatment"
            ),
            "boundary": {"llm_calls": 0, "consumer_fits": 0,
                         "held_out_reads": 0},
        }

    activations = _activations(course)
    cards = []
    for skill_id in active:
        evidence = [row for row in activations
                    if skill_id in (row.get("active_skill_ids") or ())]
        gate_ok = bool(evidence) and all(
            (row.get("delayed") or {}).get("gate", {}).get("passes") is True
            and (row.get("gate_disagreement") or {}).get("resolved_by")
            == "p4_gate"
            and (row.get("gate_disagreement") or {}).get("may_activate") is True
            for row in evidence)
        support_ok = bool(evidence) and all(row.get("probes")
                                           for row in evidence)
        threshold = {"passed": True, "clauses": 0}
        for row in evidence:
            threshold = _clause_thresholds_are_frozen_edges(
                row.get("serving_scope"))
            if not threshold["passed"]:
                break
        replay_ok = all(str(row.get("deployed_via")) != "replay"
                        for row in evidence)
        card = {
            "skill_id": skill_id,
            "activations": len(evidence),
            "passed_the_authoritative_gate": bool(gate_ok and support_ok),
            "has_a_support_reading": support_ok,
            "threshold_came_from_the_tool": bool(threshold["passed"]),
            "threshold_detail": threshold,
            "no_replay_granted_activation": bool(replay_ok),
            "units": [row.get("unit") for row in evidence],
        }
        card["passed"] = all(card[name] for name in ASSERTIONS)
        cards.append(card)

    failed = [card for card in cards if not card["passed"]]
    return {
        "stage": "HEC1_K0_AUDIT",
        "written_at": datetime.now().astimezone().isoformat(),
        "contract_version": contract.VERSION,
        "k0_empty": False,
        "cards": cards,
        "cards_total": len(cards),
        "cards_passed": len(cards) - len(failed),
        "passed": not failed,
        "may_continue": not failed,
        "verdict": ("K0_PROVENANCE_CLEAN" if not failed
                    else "K0_PROVENANCE_FAILED"),
        "assertions": list(ASSERTIONS),
        "on_failure": "stop; a card that cannot prove its route may not be K0",
        "boundary": {"llm_calls": 0, "consumer_fits": 0, "held_out_reads": 0},
    }


def _md(payload: Mapping[str, Any]) -> str:
    if payload["k0_empty"]:
        return "\n".join([
            "# HEC-1 K0 audit",
            "",
            "**K0 is empty**, which is a legal freeze: `%s`." % payload["verdict"],
            "",
            payload["reading"] + ".",
            "",
        ]) + "\n"
    lines = [
        "# HEC-1 K0 audit",
        "",
        "Three assertions per card: the authoritative gate, the tool's "
        "threshold, and no replay-granted activation.",
        "",
        "| card | gate | threshold | no replay | verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for card in payload["cards"]:
        lines.append("| `%s` | %s | %s | %s | %s |" % (
            card["skill_id"], card["passed_the_authoritative_gate"],
            card["threshold_came_from_the_tool"],
            card["no_replay_granted_activation"],
            "PASS" if card["passed"] else "**FAIL**"))
    lines += ["", "**%s** (%d/%d cards)." % (
        payload["verdict"], payload["cards_passed"], payload["cards_total"]), ""]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("course", help="the Phase S course artifact")
    parser.add_argument("--k0", default=str(
        (ARTIFACTS / "hec1_k0.json").relative_to(PROJECT_ROOT).as_posix()))
    args = parser.parse_args(argv)
    course = json.loads(Path(args.course).read_text(encoding="utf-8"))
    k0_path = PROJECT_ROOT / args.k0
    k0 = (json.loads(k0_path.read_text(encoding="utf-8"))
          if k0_path.is_file() else {"active_skill_ids": []})
    payload = audit(course, k0)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    print("K0 empty     : %s" % payload["k0_empty"])
    print("verdict      : %s" % payload["verdict"])
    print("may continue : %s" % payload["may_continue"])
    for card in payload.get("cards") or ():
        print("  %-46s %s" % (card["skill_id"],
                             "PASS" if card["passed"] else "FAIL"))
    print("wrote %s" % OUT_JSON.relative_to(PROJECT_ROOT).as_posix())
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
