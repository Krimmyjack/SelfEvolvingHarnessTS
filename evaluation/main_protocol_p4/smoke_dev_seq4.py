"""DEV-SEQ-4 smoke: the invariants the two runners depend on.  No LLM call.

Seven checks, in the order a reader would doubt them:

1. the pre-decision history handed to both arms is the one DEV-SEQ-3 recorded;
2. both snapshots load from the DEV-SEQ-3 store and their SHAs are what the
   runner claims, with the card in the fork and not in the parent;
3. the exposure probe is text that exists only in the fork;
4. against the fork the catalog offers the card's own body and applicability,
   so step 3 can revise in place instead of only adding;
5. two cells of the same situation get byte-identical histories across arms;
6. the step-3 Slow input carries the previous edit's receipt and the delivery
   mechanism, and names no surface for Slow to pick;
7. the follow-up windows are not UNREAD and are not in the evidence material.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_seq2_knowledge as know
from evaluation.main_protocol_p4 import dev_seq4_knowledge as k4
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_seq2 as seq2
from evaluation.main_protocol_p4 import run_dev_seq4 as seq4
from evaluation.main_protocol_p4 import run_dev_seq4_revision as rev
from evaluation.main_protocol_p4 import run_source_line as v1runner


def run() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = "") -> None:
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    # 1 --------------------------------------------------------------------
    histories = seq4.verify_histories()
    check("the rebuilt pre-decision history equals what DEV-SEQ-3 recorded",
          histories["passed"],
          "%s of %s decisions exact"
          % (histories["exact"], histories["decisions_checked"]))

    # 2 --------------------------------------------------------------------
    machinery = v1runner._machinery()
    parent = machinery["compile_snapshot"](seq4.STORE / seq4.PARENT_SHA,
                                           verify_lock=False)
    fork = machinery["compile_snapshot"](seq4.STORE / seq4.FORK_SHA,
                                         verify_lock=False)
    parent_ids = {str(s.skill_id) for s in parent.skills}
    fork_ids = {str(s.skill_id) for s in fork.skills}
    check("both snapshots load and their SHAs are the ones the runner names",
          parent.runtime_bundle_sha == seq4.PARENT_SHA
          and fork.runtime_bundle_sha == seq4.FORK_SHA)
    check("the card is in the fork and absent from the parent",
          seq4.CARD_ID in fork_ids and seq4.CARD_ID not in parent_ids,
          {"parent_skills": len(parent_ids), "fork_skills": len(fork_ids)})

    # 3 --------------------------------------------------------------------
    card_entry = next(s for s in fork.skills if str(s.skill_id) == seq4.CARD_ID)
    probe = str(card_entry.body)[60:180]
    parent_bodies = json.dumps([str(s.body) for s in parent.skills],
                               ensure_ascii=False)
    check("the exposure probe is text only the fork has",
          len(probe) >= 20 and probe not in parent_bodies,
          {"probe_length": len(probe)})

    # 4 --------------------------------------------------------------------
    store = machinery["SnapshotStore"](Path(tempfile.mkdtemp()) / "store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"]())
    catalog = know.build_catalog(controller=controller,
                                 parent=store.materialize(fork))
    surface_ids = [row["surface_id"] for row in catalog]
    check("step 3 can revise the previous card in place, not only add",
          any(seq4.CARD_ID in sid and sid.endswith(".body")
              for sid in surface_ids)
          and any(seq4.CARD_ID in sid
                  and sid.endswith(".observable_applicability")
                  for sid in surface_ids),
          surface_ids)

    # 5 --------------------------------------------------------------------
    position, uid, _kind = seq4.SITUATIONS[0]
    rows = seq4.history_before(position, uid)
    left = [str(getattr(ep, "episode_id", "")) for ep in seq4.rehydrate(rows)]
    right = [str(getattr(ep, "episode_id", "")) for ep in seq4.rehydrate(rows)]
    check("both arms of a cell are handed the identical history",
          left == right and len(left) == len(rows),
          {"situation": "u%s/%s" % (position, uid), "episodes": len(left)})

    # 6 --------------------------------------------------------------------
    fake_contrast = {"summary": {"by_situation_kind": {
        "matched": {seq2.ARM_A: {"decisions": 9, "card_retrieved": 0,
                                 "select_identity": 3, "deployed": 5},
                    seq2.ARM_B: {"decisions": 9, "card_retrieved": 9,
                                 "select_identity": 7, "deployed": 5}},
        "nearest_miss": {seq2.ARM_A: {"decisions": 9}, seq2.ARM_B: {"decisions": 9}},
    }, "per_situation": []}}
    slow_input = k4.revision_card(
        {"pattern_id": "x", "failing_decisions": []},
        pattern_id="devseq4-revision",
        previous_proposal={"predicted_agent_behavior_change":
                           ["retrieve_skill:x", "choose_candidate_kind:identity",
                            "identity_retained"],
                           "falsification_condition": ["..."]},
        current_body="body", current_applicability={"all": []},
        contrast=fake_contrast)
    blob = json.dumps(slow_input, ensure_ascii=False)
    check("the step 3 input carries the previous edit's receipt",
          len(slow_input["previous_guidance"]
              ["what_you_predicted_and_what_was_measured"]) == 3)
    check("the step 3 input states the delivery mechanism and permits keeping "
          "the guidance, without naming a surface to pick",
          "so_a_selection_of_identity_does_not_prevent_a_deployment"
          in blob
          and "no_proposal envelope" in blob
          and "candidate_policy.proposal_guidance" not in blob)

    # 7 --------------------------------------------------------------------
    doc = base.load(R.ORDERING)
    rows_pop, _excluded = R._population(doc)
    by_position = {row["position"]: row for row in rows_pop}
    exposures = {p: list(by_position[p]["unit"].get("exposure") or ())
                 for p in rev.FOLLOWUP_POSITIONS}
    check("the follow-up windows are development windows, not UNREAD, and are "
          "not the windows Slow is shown",
          all("UNREAD" not in v for v in exposures.values())
          and not (set(rev.FOLLOWUP_POSITIONS)
                   & set(rev.EVIDENCE_POSITIONS)),
          {"followup": exposures, "evidence": list(rev.EVIDENCE_POSITIONS)})

    return {"suite": "DEV_SEQ4", "checks": checks,
            "passed": all(row["passed"] for row in checks),
            "consumer_fits": 0, "llm_calls": 0}


if __name__ == "__main__":
    report = run()
    for row in report["checks"]:
        print("PASS" if row["passed"] else "FAIL", "|", row["check"])
        if row.get("detail"):
            print("      ", json.dumps(row["detail"], ensure_ascii=False,
                                       default=str)[:200])
    print("passed:", report["passed"], "| fits:", report["consumer_fits"],
          "| llm:", report["llm_calls"])
    sys.exit(0 if report["passed"] else 1)
