"""Layer 1: one real outer step -- forward, A5-online, k=1 -- through the
repaired path, with a live Slow and a real replay screen.

Authorised ceiling: **2 physical Slow calls, 10 Consumer fits.**  Both are
enforced before the spend, not checked after it:

* ``BudgetGuard(ordering_cap=2)`` refuses the third request inside
  ``_MeteredOuterBackend.complete`` -- before the transport -- and the inner
  ``BudgetedAgentBackend(maximum_calls=2)`` refuses it again.
* ``OuterBudget(replay_fits_remaining=10)`` makes ``consolidate`` compare a
  screen's *published worst case* against what is left and refuse the screen
  whole rather than start one it cannot finish.  A guard on the callable
  refuses again if the ledger has already reached the ceiling.

Nothing about the method, the thresholds, the candidate space or the
information wall is changed here.  The step is rebuilt from the frozen course
record: the same bank rows, the same Draft in the same lifecycle state, the
same Active lineage, the same contract constants.  This runner only supplies
the two things the historical step could not get past -- a Slow that is asked,
and a screen that is paid for -- and then records exactly where it stops.

Not layer 2: no later unit is touched, nothing is verified on a new unit, and
no Skill is activated.  ``record_revision`` writing a clause onto the Draft is
the end of this layer.

Run:  python -m evaluation.main_protocol_p4.run_m_r0d_forward_k1_outer_step
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import audit_m_r0b_revision_opportunity as opp
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_source_line as v1runner

ART = base.ART
ORDERING = "forward"
ARM = "A5-online"
K_INDEX = 1
BOUNDARY_POSITION = 4

#: The authorised ceiling.  Not a target and not a contract change: the
#: protocol's own OUTER_LLM_PER_STEP is already 2, and 10 is what one replay
#: screen over this step's five processed cells costs at the cached rate.
MAX_SLOW_CALLS = 2
MAX_CONSUMER_FITS = 10


class FitCeiling(RuntimeError):
    """The authorised fit ceiling would be crossed.  Refused before spending."""


def _state_at_k1(doc: dict[str, Any]) -> dict[str, Any]:
    """The arm's bank, processed cells, Draft and Active set at the boundary."""
    bank: list[dict[str, Any]] = []
    processed = []
    held: set[str] = set()
    k0 = json.loads(base.K0_RECEIPT.read_text(encoding="utf-8"))
    held.update(str(key) for key in (k0.get("lineage_keys") or ()))

    for cell in base.cells(doc, ARM):
        position = int(cell["position"])
        if position > BOUNDARY_POSITION:
            break
        if cell.get("lineage_key"):
            held.add(str(cell["lineage_key"]))
        for row in base.bank_rows_for_cell(cell):
            bank.append(opp._bank_row(cell, row))
        processed.append(opp._unit_ctx(cell["unit"]))

    timelines = opp._draft_timeline(doc, ARM)
    ledger = drafts.DraftLedger()
    reconstructed = []
    for timeline in timelines:
        state = opp._draft_state_at(timeline, BOUNDARY_POSITION)
        if not state.get("exists"):
            continue
        key = outer_loop.census_key(base.TASK_CONSUMER_KEY,
                                    [{"op": op, "params": {}}
                                     for op in timeline["program"].split(">")],
                                    timeline["root_scope"])
        draft = drafts.RestrictedDraft(
            draft_id=timeline["draft_id"],
            program_steps=tuple((op, {})
                                for op in timeline["program"].split(">")),
            root_scope=dict(timeline["root_scope"]),
            current_scope=dict(timeline["root_scope"]),
            revisions=0,
            created_at_origin=int(timeline["events"][0].get("window") or 0),
            census_key=key,
        )
        draft.state = state["state"]
        draft.verification_attempts = int(state["verification_attempts"])
        draft.history.append({"event": "verification",
                              "state_after": state["state"]})
        ledger.drafts.append(draft)
        reconstructed.append({
            "draft_id": draft.draft_id, "program": timeline["program"],
            "state": draft.state, "revisions": draft.revisions,
            "verification_attempts": draft.verification_attempts,
            "census_key": key, "current_scope": dict(draft.current_scope),
        })
    return {"bank": bank, "processed": processed, "held": sorted(held),
            "ledger": ledger, "drafts_before": reconstructed, "k0": k0}


def _live_slow(guard: runner.BudgetGuard, snapshot: Any):
    machinery = v1runner._machinery()
    target = machinery["agentic"].live_transport()
    inner = machinery["agentic"]._default_backend_factory(MAX_SLOW_CALLS)
    metered = runner._MeteredOuterBackend(inner, guard=guard, billable=True)
    core = machinery["TTHAAgentCore"](
        metered,
        machinery["LocalPublicToolGateway"](np.zeros(8, dtype=np.float64),
                                            task_kind="forecast"),
        model=target["model"], base_url=target["base_url"])
    agent = runner.OuterSlowAgent(
        core, vocabulary=contract.SCOPE_CLASS["vocabulary"],
        guard=guard, snapshot=snapshot)
    return agent, metered, target


def _capped_replay(processed, ledgers: runner.Ledgers,
                   cache: runner.ReplayPredictionCache):
    inner = runner.replay_screen_for(processed, ledgers, cache)

    def replay(*, steps, scope):
        if ledgers.replay_fits >= MAX_CONSUMER_FITS:
            raise FitCeiling(
                "the authorised ceiling of %d Consumer fits is already spent"
                % MAX_CONSUMER_FITS)
        return inner(steps=steps, scope=scope)

    replay.estimated_fits_per_candidate = getattr(
        inner, "estimated_fits_per_candidate", 0)
    return replay


def build() -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    doc = base.load(ORDERING)
    state = _state_at_k1(doc)

    machinery = v1runner._machinery()
    k0 = state["k0"]
    # The snapshot the *forward course* recorded starting from, not the Phase S
    # receipt's own store.  Both name the same runtime_bundle_sha -- it is one
    # snapshot under two paths -- but the Phase S store is not readable on this
    # machine, and the course's own path is the one Phase T actually compiled.
    course_k0 = dict(doc.get("k0_snapshot") or {})
    candidates = [
        (course_k0.get("store_root"), course_k0.get("runtime_bundle_sha"),
         "course artifact k0_snapshot"),
        (k0.get("store_root"), k0.get("runtime_bundle_sha"),
         "Phase S K0 receipt"),
    ]
    snapshot_dir = None
    snapshot_source = None
    for store_root, sha, label in candidates:
        if not (store_root and sha):
            continue
        path = base.ROOT / str(store_root) / str(sha)
        try:
            readable = path.is_dir() and (path / "snapshot.lock.json").exists()
        except OSError:
            readable = False
        if readable:
            snapshot_dir, snapshot_source = path, label
            break
    if snapshot_dir is None:
        return {"status": "BLOCKED", "stage": "M_R0D_FORWARD_K1_OUTER_STEP_LIVE",
                "why": "no readable K0 snapshot among %s"
                       % [str(row[0]) for row in candidates],
                "actual_cost": {"physical_slow_calls": 0,
                                "consumer_fits_replay": 0}}
    if (course_k0.get("runtime_bundle_sha")
            and k0.get("runtime_bundle_sha")
            and course_k0["runtime_bundle_sha"] != k0["runtime_bundle_sha"]):
        return {"status": "BLOCKED", "stage": "M_R0D_FORWARD_K1_OUTER_STEP_LIVE",
                "why": ("the course and the receipt name different K0 bundles; "
                        "refusing to start from either"),
                "actual_cost": {"physical_slow_calls": 0,
                                "consumer_fits_replay": 0}}
    snapshot = machinery["compile_snapshot"](snapshot_dir, verify_lock=False)

    ledgers = runner.Ledgers()
    guard = runner.BudgetGuard(
        ordering_cap=MAX_SLOW_CALLS,
        per_unit_arm_cap=int(contract.PER_UNIT_ARM_BUDGET["llm_calls"]),
        ledgers=ledgers)
    slow, metered, target = _live_slow(guard, snapshot)
    cache = runner.ReplayPredictionCache(ARM)
    replay = _capped_replay(state["processed"], ledgers, cache)

    budget = outer_loop.OuterBudget(
        outer_llm_per_step=int(contract.OUTER_LLM_PER_STEP),
        replay_fits_remaining=MAX_CONSUMER_FITS)

    fault = None
    try:
        record = outer_loop.consolidate(
            bank=state["bank"], ledger=state["ledger"], k_index=K_INDEX,
            slow=slow, replay=replay, budget=budget,
            held_lineage_keys=state["held"],
            bank_boundary={"arm": ARM,
                           "units": [ctx.unit for ctx in state["processed"]],
                           "excludes": ["the evaluation face (+144)",
                                        "future units", "other arms",
                                        "held-out"]})
        step = record.to_dict()
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        fault = "%s: %s" % (type(exc).__name__, str(exc)[:400])
        step = None

    after = [{"draft_id": draft.draft_id,
              "state": draft.state,
              "revisions": draft.revisions,
              "verification_attempts": draft.verification_attempts,
              "closed": draft.closed,
              "census_key": draft.census_key,
              "current_scope": dict(draft.current_scope),
              "root_scope": dict(draft.root_scope),
              "clauses_added_since_root": draft.clauses_added_so_far(),
              "revision_history": [dict(row) for row in draft.revision_history],
              "history_events": [row.get("event") for row in draft.history],
              "deployable": draft.deployable}
             for draft in state["ledger"].drafts]

    before = state["drafts_before"]
    target_draft = next((row for row in after
                         if row["draft_id"] == "resupplied_draft_1"), None)
    before_draft = next((row for row in before
                         if row["draft_id"] == "resupplied_draft_1"), None)

    return {
        "stage": "M_R0D_FORWARD_K1_OUTER_STEP_LIVE",
        "layer": "1 of 2 -- one outer step; no later unit is touched",
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "authorised_ceiling": {
            "slow_calls": MAX_SLOW_CALLS,
            "consumer_fits": MAX_CONSUMER_FITS,
            "enforced": "before the spend, by the guard and the outer budget",
        },
        "target": {"ordering": ORDERING, "arm": ARM, "k_index": K_INDEX,
                   "boundary_position": BOUNDARY_POSITION,
                   "units_in_bank": len(state["processed"]),
                   "bank_rows": len(state["bank"]),
                   "held_lineage_keys": state["held"]},
        "transport": {"base_url": target["base_url"], "model": target["model"],
                      "source": target["source"]},
        "k0_snapshot": {"path": str(snapshot_dir.relative_to(base.ROOT)),
                        "resolved_from": snapshot_source,
                        "runtime_bundle_sha": str(
                            course_k0.get("runtime_bundle_sha")
                            or k0.get("runtime_bundle_sha"))},
        "code_state": base.version_check(),
        "drafts_before": before,
        "drafts_after": after,
        "state_written": {
            "revisions": ((before_draft or {}).get("revisions"),
                          (target_draft or {}).get("revisions")),
            "current_scope_changed": (
                (before_draft or {}).get("current_scope")
                != (target_draft or {}).get("current_scope")),
            "verification_attempts_unchanged": (
                (before_draft or {}).get("verification_attempts")
                == (target_draft or {}).get("verification_attempts")),
            "state_unchanged": ((before_draft or {}).get("state")
                                == (target_draft or {}).get("state")),
            "no_second_shell": len(after) == len(before),
            "still_not_deployable": all(not row["deployable"] for row in after),
        },
        "outer_step": step,
        "run_fault": fault,
        "slow_calls_recorded_by_the_agent": slow.calls,
        "actual_cost": {
            "physical_slow_calls": int(metered.calls),
            "billed_llm_outer": ledgers.llm_outer,
            "billed_llm_fast": ledgers.llm_fast,
            "consumer_fits_replay": ledgers.replay_fits,
            "replay_cache": cache.to_dict(),
            "within_ceiling": (int(metered.calls) <= MAX_SLOW_CALLS
                               and ledgers.replay_fits <= MAX_CONSUMER_FITS),
            "blocked_before_backend": guard.blocked,
        },
        "boundary": {
            "later_units_touched": 0,
            "skills_activated": 0,
            "held_out_reads": 0,
            "sealed_reads": 0,
            "evaluation_face_reads": 0,
            "thresholds_changed": 0,
            "production_code_changed": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    # A label per attempt.  The first attempt's receipt is an instrument record
    # of a transport refusal and stays exactly as written; a retry gets its own
    # file rather than erasing the evidence that the first one failed.
    args = list(argv if argv is not None else sys.argv[1:])
    label = args[args.index("--label") + 1] if "--label" in args else ""
    report = build()
    report["run_label"] = label or "attempt1"
    out = ART / ("m_r0d_forward_k1_outer_step_live.json" if not label
                 else "m_r0d_forward_k1_outer_step_live__%s.json" % label)
    if out.exists():
        sys.stderr.write("refusing to overwrite %s\n" % out)
        return 2
    # A receipt must survive whatever the live path hands back.  Attempt 2 spent
    # a real Slow call, got an answer, and then died here: the validated stage
    # payload comes back as a frozen ``mappingproxy``, which ``json.dumps``
    # refuses, and the whole record was lost *after* the budget was spent.
    # ``drafts._plain`` already exists for exactly this shape; ``default=str``
    # catches anything else rather than letting it destroy the evidence again.
    report = drafts._plain(report)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False,
                              default=str),
                   encoding="utf-8")
    print("wrote %s" % out)
    print(json.dumps({k: report[k] for k in
                      ("actual_cost", "state_written", "run_fault")
                      if k in report}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
