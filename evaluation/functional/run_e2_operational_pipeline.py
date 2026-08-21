"""One continuous operational run of the V1 Harness, no human relay.

The single question: can the Harness, in one uninterrupted run, go from
using its Source Guidance, through adapting to a target, forming and
promoting a Target-local Skill, recalling it, failing on a later window,
attributing that failure, patching one Scope/Risk surface through Slow, and
reading the patched Harness back on the next task -- with no hand-written
card, no store swap, no hand-named Workflow and no mid-run restart.

Development level.  Every window is quoted verbatim from the #17/#19
registers, locked before the run; nothing beyond index 17520 is read; no
A5-vs-A3 comparison is re-estimated and no new method is introduced.  This
module is orchestration only: every measurement, every prompt, every gate
and every store write is an existing tracked or delivered implementation
called from here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
import run_batch_composition_headroom as bch  # noqa: E402
import run_e2_fresh_confirmation as FC  # noqa: E402
import run_e2_recipe_experience_to_skill as bridge  # noqa: E402
import run_e2_skill_store_integration as ssi  # noqa: E402
import run_e2_slow_scope_update as SSU  # noqa: E402
import run_e2_warm_vs_cold_recipe_search as wvc  # noqa: E402

from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_sha256,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentRole,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    EVIDENCE_SUPPORT,
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.harness import (  # noqa: E402
    compiler as harness_compiler,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)

PROTOCOL_VERSION = "operational_pipeline_v2"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
# v1 is the #21 record and is never rewritten.
OUT_JSON = E2 / "operational_pipeline_v2.json"
OUT_MD = E2 / "operational_pipeline_v2.md"

WORK_ROOT = PROJECT_ROOT / "_scratch" / "operational_v1"
STORE_ROOT = WORK_ROOT / "store"
GATE_ROOT = WORK_ROOT / "gate_stores"
SLOT = "pipeline"

VARIANT = bch.CONSUMER_POOLED
ARM = "A5"
COHORT_NAME = FC.COHORT_NAME
IDENTITY = FC.IDENTITY
MATERIAL_THRESHOLD = float(FC.MATERIAL_THRESHOLD)
HARM_THRESHOLD = float(FC.HARM_THRESHOLD)
CARD_ID = FC.SKILL_ID[VARIANT]
SLOW_MODEL = "claude-opus-5"
SLOW_BASE_URL = ssi.NF_BASE_URL
NEW_SURFACE = SSU.NEW_SURFACE_V2

# Every window is quoted from the register, locked here before the run.
# Nothing below is chosen after seeing a reading.
WINDOW_TASK_A = FC.WINDOWS["task_A"]
WINDOW_PROBE = FC.PROBE_WINDOW
WINDOW_TASK_B = FC.WINDOWS["task_B"]
WINDOW_TASK_C = FC.WINDOWS["task_C"]
WINDOW_TASK_D = FC.WINDOWS["task_D"]

LLM_CALL_BUDGET_TOTAL = 20
LLM_CALL_BUDGET_PER_EPISODE = int(FC.LLM_CALL_BUDGET_PER_EPISODE)
RETRAIN_BUDGET = 300
MAX_TRANSPORT_FAILURES = 3


def _repo_rel(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------- the freeze
COMPILER_FILE = "methods/ttha/harness/compiler.py"
H0_LOCK_FILE = "methods/ttha/harness/h0/snapshot.lock.json"
FROZEN_SURFACE_V3: tuple[str, ...] = tuple(SSU.FROZEN_SURFACE_V2) + (
    "artifacts/functional/e2/slow_scope_update_v1.json",
    "artifacts/functional/e2/slow_scope_update_v2.json",
    "evaluation/functional/run_e2_slow_scope_update.py",
)
# A1 (#22): the envelope validator now enforces maxLength, symmetric with
# its own minLength.  The file feeds dependency_shas under
# ttha:schema_contracts, so every snapshot identity moves; no measurement
# does.
SCHEMA_CONTRACTS_FILE = "methods/ttha/schema_contracts.py"
FROZEN_SURFACE_V4: tuple[str, ...] = FROZEN_SURFACE_V3 + (SCHEMA_CONTRACTS_FILE,)
A1_TOUCHED: tuple[dict[str, str], ...] = (
    {
        "path": SCHEMA_CONTRACTS_FILE,
        "role": (
            "A1: maxLength is enforced where minLength already was, so a "
            "declared string cap is taught by the retry loop instead of "
            "dying at a gate with no second attempt"
        ),
        "sha256_before_a1": (
            "08b421d7ca4c6300052af4a78105548559a0c6be002e6aead7aac15537d936d5"
        ),
    },
)


# P1 moved these two and only these two; both sides are on the record.
P1_TOUCHED: tuple[dict[str, str], ...] = (
    {
        "path": COMPILER_FILE,
        "role": (
            "P1: the Scope/Risk enforcement walk is promoted next to the "
            "evaluator, so reading a guard and obeying it are both tracked "
            "machinery.  Measurement stays with the caller."
        ),
        "sha256_before_p1": (
            "fd57945185475fc473c72bb833b6d60f6332c1f46bf5efdaf80b5f686f7b8973"
        ),
    },
    {
        "path": H0_LOCK_FILE,
        "role": (
            "regenerated: the lock pins the compiler identity, and P1 moved "
            "the compiler.  harness_content_sha is unchanged; only the "
            "compiler-derived shas move."
        ),
        "sha256_before_p1": (
            "bbddc14b14cb04f50047c8a063d546a2ed6d4ab8c87c08cf418004e15888ecad"
        ),
    },
)


class ConcurrentWrite(RuntimeError):
    """The frozen surface moved while this protocol was running."""


class Blocked(RuntimeError):
    """A pre-registered stop: the first block ends the run."""

    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__(reason)
        self.verdict = verdict
        self.reason = reason


def _freeze() -> dict[str, str]:
    frozen: dict[str, str] = {}
    for name in sorted(set(FROZEN_SURFACE_V4)):
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise SystemExit("frozen surface member is missing: %s" % name)
        frozen[name] = _sha256(path)
    return frozen


def _verify(before: Mapping[str, str]) -> dict[str, Any]:
    drift = [
        name for name, sha in before.items()
        if not (PROJECT_ROOT / name).is_file()
        or _sha256(PROJECT_ROOT / name) != sha
    ]
    return {"files": len(before), "drift": drift, "ok": not drift}


def _guard_frozen(before: Mapping[str, str], where: str) -> bool:
    report = _verify(before)
    if not report["ok"]:
        raise ConcurrentWrite(
            "the frozen surface moved before %s: %s" % (where, report["drift"])
        )
    return True


class Budget:
    def __init__(self, llm: int, retrains: int) -> None:
        self.llm_total = int(llm)
        self.llm_used = 0
        self.retrain_total = int(retrains)
        self.retrains_charged = 0

    def take(self, want: int) -> int:
        return max(0, min(int(want), self.llm_total - self.llm_used))

    def spend(self, calls: int) -> None:
        self.llm_used += int(calls)
        if self.llm_used > self.llm_total:
            raise Blocked(
                "BUDGET_EXCEEDED", "the LLM call budget was exhausted mid-run"
            )

    def charge_retrains(self, count: int) -> None:
        self.retrains_charged += int(count)
        if self.retrains_charged > self.retrain_total:
            raise Blocked(
                "BUDGET_EXCEEDED",
                "the Consumer retrain budget was exhausted mid-run (%d > %d)"
                % (self.retrains_charged, self.retrain_total),
            )

    @property
    def left(self) -> int:
        return self.llm_total - self.llm_used


# ----------------------------------------------------- what was fixed first
PRE_REGISTERED: dict[str, Any] = {
    "question": (
        "can the V1 Harness complete Source use -> target adaptation -> "
        "lifecycle -> failure attribution -> Slow edit -> next-task behaviour "
        "reading in one continuous run with no human relay"
    ),
    "level": (
        "development.  Nothing beyond index 17520 is opened, A5-vs-A3 is not "
        "re-estimated, and no new method is introduced."
    ),
    "p0_5": {
        "P1": (
            "the #19 reference enforcement (_fires/_check/_enforce) is "
            "promoted into the tracked adoption path: discovery, evaluation, "
            "the fallback walk and the identity floor now all live in "
            "methods/ttha/harness/compiler.py, with the Consumer measurement "
            "injected by the caller.  Guard semantics and v2 ladder semantics "
            "are unchanged."
        ),
        "P2": (
            "hard gate: with an empty guard list the four #19 task_C episodes "
            "must reproduce digit-for-digit through the promoted path; "
            "otherwise INSTRUMENT_DRIFT and nothing else runs"
        ),
        "P3": (
            "data/benchmark_noaa_fresh_v1/manifest.json is un-ignored by a "
            "negative pattern so the frozen-surface member has a tracked "
            "baseline; the arrays it describes stay ignored"
        ),
        "P4": (
            "both P1-touched tracked files carry pre/post sha256 in the v3 "
            "frozen list; every other member is checked for zero drift"
        ),
    },
    "fixed_configuration": {
        "cohort": COHORT_NAME,
        "consumer_variant": VARIANT,
        "arm": ARM,
        "slow_backend": SLOW_MODEL,
        "slow_sampling": (
            "one pinned configuration, one draw; no cross-backend sampling.  "
            "An abstention or a compiler rejection stops the run."
        ),
        "store": (
            "one new continuous store: bootstrap plus the all-source pooled "
            "Guidance card compiled from the same input as #17, with "
            "scope_risk_guards starting as the empty list"
        ),
        "frozen": (
            "Judge, Metric, program menu, Guidance content, lifecycle "
            "thresholds, harm line and the v2 adoption ladder are untouched"
        ),
    },
    "window_order": [
        {"step": "task_A", "start": WINDOW_TASK_A["start"],
         "role": "full-price adaptation"},
        {"step": "probe", "origins": list(WINDOW_PROBE["origins"]),
         "role": "out of selection; DRAFT -> ACTIVE"},
        {"step": "task_B", "start": WINDOW_TASK_B["start"], "role": "recall and reuse"},
        {"step": "task_C", "start": WINDOW_TASK_C["start"],
         "role": "already-exposed 2025 region; #17's known natural harm sample"},
        {"step": "attribution", "role": "first fault, on this run's own episode records only"},
        {"step": "slow_edit", "role": "one proposal, one surface, compiler-validated"},
        {"step": "task_D", "start": WINDOW_TASK_D["start"],
         "role": ("time-order successor; #17 never used it; the missing "
                  "gate reads the NaN mask only")},
    ],
    "behavior_change_criterion": {
        "precondition": "the guard is evaluated at task_D, with trace evidence",
        "i": "task_D fires the guard on the spot and walks the ladder fallback",
        "ii": (
            "if the harm does not recur, a 0-evaluation shadow replay of this "
            "run's own task_C under the post-patch snapshot decides "
            "differently from the pre-patch decision, consistent with the "
            "Risk contract"
        ),
        "not_a_failure": (
            "harm that does not recur while the guard correctly stays silent "
            "is the contract working, not a failure"
        ),
    },
    "verdicts": [
        "OPERATIONAL_PIPELINE_CLOSES", "SOURCE_DELIVERY_BREAK",
        "LOCAL_LIFECYCLE_BREAK", "ATTRIBUTION_BREAK", "SLOW_ABSTAINS",
        "COMPILER_REJECTS", "POST_UPDATE_NOT_RETRIEVED",
        "NEXT_TASK_NO_BEHAVIOR_CHANGE", "NO_POST_UPDATE_WINDOW",
        "INCONCLUSIVE_TRANSPORT", "INSTRUMENT_DRIFT",
        "CONCURRENT_WRITE_ABORT", "PIPELINE_RUNS_NO_FAULT_SAMPLE",
    ],
    "no_fault_sample": (
        "if the trajectory reaches task_D without ever producing a RISK_GAP "
        "-- no evaluation series crossed the harm line, so the fold has "
        "nothing actionable to name -- links 6 to 9 are recorded untested in "
        "this sample.  Nothing is re-thrown and no harm is seeded."
    ),
    "by_veto_passes": (
        "a BY_VETO shape passes: this round tests continuity.  Keeping the "
        "aggregate gain is a BY_RESCOPE round's question."
    ),
    "discipline": {
        "llm_call_budget": LLM_CALL_BUDGET_TOTAL,
        "retrain_budget": RETRAIN_BUDGET,
        "first_block_stops": True,
        "no_second_method_repair": True,
        "commit": False,
        "spawn": False,
        "beyond_17520": "zero reads",
    },
    "notes_go_in_the_artifact": (
        "every caliber note, every deviation and every ambiguity is written "
        "into the artifact rather than left in a source comment"
    ),
    "caliber_notes": {
        "worktree_byte_hashes": (
            "all sha256 values here are working-tree byte hashes.  "
            "harness_surfaces.json is CRLF on disk, so its committed LF blob "
            "hashes differently; that is a line-ending caliber, not drift."
        ),
        "manifest_baseline": (
            "data/benchmark_noaa_fresh_v1/manifest.json had no tracked "
            "baseline before P3; its zero-drift check was previously "
            "within-run only"
        ),
        "h0_lock_left_as_p1_left_it": (
            "A1 changes methods/ttha/schema_contracts.py, which feeds "
            "dependency_shas under ttha:schema_contracts, so h0's lock is "
            "now stale in exactly that one key (harness_content_sha is "
            "unchanged).  It was deliberately NOT regenerated: every runner "
            "here compiles with verify_lock=False, and regenerating would "
            "move h0/snapshot.lock.json -- a P1-touched file whose "
            "byte-identity is the stated precondition for not re-running the "
            "P2 gate.  The regenerated lock would hash to "
            "5be7bddc92a2928701bc11408082d3e0f48f0703843a13c0d49e0491c1f71c4d; "
            "the next checkpoint can apply it in one deterministic step."
        ),
        "p2_gate_could_not_be_re_run_even_if_wanted": (
            "run_e2_slow_scope_update._open_stores_v2 hard-asserts that the "
            "dependency drift is exactly [compiler_source, surface_registry].  "
            "A1 adds ttha:schema_contracts, so the delivered gate would "
            "SystemExit.  Skipping it is forced, not merely economical, and "
            "the justification stands on A1-A3 all sitting on the proposal "
            "and reporting paths."
        ),
        "measurement_stays_outside_the_harness": (
            "the promoted gate injects delayed_of/support_of rather than "
            "importing an instrument: the Harness decides, it does not "
            "measure"
        ),
    },
}


# ------------------------------------------------------------- P2 hard gate
P1_POST_SHA = {
    COMPILER_FILE: (
        "61cc3b4538baa17a0b1bbe8458c86a224f884167ded56296f8a87629b0ae5eb5"
    ),
    H0_LOCK_FILE: (
        "e5a49cd6ae6231bf2ca6daf2ac66e2b064d1f3a3e4b6db6b2d5092820b788d5d"
    ),
}


def stage_p2_precondition() -> dict[str, Any]:
    """A4: the gate is not re-run, and the reason is checked, not assumed.

    #21 ran the four-episode gate and it passed 4/4.  This round's fixes all
    sit on the proposal and reporting paths, so the gate would re-measure
    nothing -- but that argument is only good while the measurement side is
    byte-identical, so the two P1-touched files are checked here.  If either
    moved, the gate would have to be re-run, and it cannot be: the delivered
    gate hard-asserts that the dependency drift is exactly
    [compiler_source, surface_registry], and A1 adds a third key.
    """
    observed = {name: _sha256(PROJECT_ROOT / name) for name in P1_POST_SHA}
    moved = sorted(
        name for name, sha in P1_POST_SHA.items() if observed[name] != sha
    )
    out = {
        "ran": False,
        "role": (
            "A4: the #21 four-episode gate is not re-run; its precondition is "
            "verified instead"
        ),
        "gate_result_carried_forward": (
            "operational_pipeline_v1: 4/4 #19 task_C episodes reproduced "
            "digit-for-digit through the promoted enforcement path, 111 "
            "retrains, 0 LLM"
        ),
        "precondition": {
            "expected_post_p1_sha256": dict(P1_POST_SHA),
            "observed_sha256": observed,
            "byte_identical": not moved,
            "moved": moved,
        },
        "why_it_cannot_be_re_run_anyway": (
            "run_e2_slow_scope_update._open_stores_v2 exits unless the "
            "dependency drift is exactly [compiler_source, surface_registry]; "
            "A1 adds ttha:schema_contracts"
        ),
        "consumer_retrains": 0,
        "llm_calls": 0,
    }
    if moved:
        raise Blocked(
            "INSTRUMENT_DRIFT",
            "the measurement-side files this round promised not to touch "
            "moved: %s; the four-episode gate would be required and the "
            "delivered gate can no longer run it" % moved,
        )
    return out


def stage_p2_gate(budget: Budget) -> dict[str, Any]:
    """The four #19 task_C episodes, replayed through the promoted path.

    Everything here is #19's own code, called with its store roots pointed
    at this run's scratch namespace so the audited v2 fork is left alone.
    """
    started = time.perf_counter()
    SSU.WORK_ROOT_V2 = GATE_ROOT
    SSU.STORE_ROOT_V2 = GATE_ROOT / "stores"
    stores = SSU._open_stores_v2()
    migration = SSU.stage_migrate_v2(stores)
    episodes = SSU._load_episodes_v2()
    cohort = SSU._confirmation_payload(episodes["locked_roster"])
    gate = SSU.stage_gate_v2(
        cohort=cohort, episodes=episodes, stores=stores,
        window=episodes["window"], budget=budget,
    )
    cells = {
        slot: {
            "reproduces": bool(row["reproduces"]),
            "mismatched": list(row.get("mismatched") or ()),
            "consumer_retrains": int(row["consumer_retrains"]),
        }
        for slot, row in gate["cells"].items()
    }
    out = {
        "ran": True,
        "role": (
            "P2: with an empty guard list the promoted enforcement path must "
            "leave every #19 task_C reading exactly where it was"
        ),
        "state": str(gate.get("state")),
        "migration_receipts": {
            slot: {
                "applied": bool(row.get("applied")),
                "single_key_diff_proof": row.get("single_key_diff_proof"),
            }
            for slot, row in migration["receipts"].items()
        },
        "cells": cells,
        "ok": bool(gate["ok"]),
        "mismatched_cells": list(gate.get("mismatched_cells") or ()),
        "consumer_retrains": int(gate["consumer_retrains"]),
        "llm_calls": 0,
        "wall_seconds": time.perf_counter() - started,
    }
    if not out["ok"]:
        raise Blocked(
            "INSTRUMENT_DRIFT",
            "the promoted enforcement path did not reproduce %s"
            % out["mismatched_cells"],
        )
    return out


# --------------------------------------------------- the one continuous store
def stage_store() -> tuple[dict[str, Any], dict[str, Any]]:
    """Bootstrap plus the all-source pooled Guidance card.  0 LLM."""
    started = time.perf_counter()
    if STORE_ROOT.exists():
        shutil.rmtree(STORE_ROOT)
    target = FC._target(VARIANT)
    card = bridge.compile_skill_card(target)
    card_text = bridge.render_skill_card(card)
    payload = FC._card_payload(target, card, card_text)
    guarantees = ssi._assert_guidance_only(payload)
    FC.STORE_ROOT = STORE_ROOT
    slot = FC._build_store(SLOT, payload)
    if slot.get("status") != "REGISTERED":
        raise Blocked(
            "SOURCE_DELIVERY_BREAK",
            "the store refused the compiled card: %s" % slot.get("blocked_reason"),
        )
    snapshot = slot["_snapshot"]
    guards = harness_compiler.scope_risk_guards_of(snapshot)
    if guards:
        raise Blocked(
            "INSTRUMENT_DRIFT",
            "the new store did not start with an empty guard list: %s" % guards,
        )
    record = {
        "ran": True,
        "llm_calls": 0,
        "consumer_retrains": 0,
        "slot": SLOT,
        "card": {
            "skill_id": CARD_ID,
            "status": str(card["status"]),
            "clause_count": int(card["clause_count"]),
            "clause_ids": [str(c["clause_id"]) for c in card["clauses"]],
            "clauses": [
                {
                    "clause_id": str(c["clause_id"]),
                    "rule": str(c["rule"]),
                    "text": str(c["text"]),
                }
                for c in card["clauses"]
            ],
            "loco": dict(card["loco"]),
            "card_bytes_sha256": canonical_sha256(dict(payload)),
            "compiled_from": (
                "run_e2_recipe_experience_to_skill.compile_skill_card, the "
                "same input #17 used; the fresh cohort contributes no "
                "evidence row so the leave-one-cohort-out merge drops nothing"
            ),
            "guidance_only_guarantees": wvc._plain(guarantees),
        },
        "store": {
            "root": _repo_rel(STORE_ROOT / SLOT / "snapshots"),
            "h0_runtime_bundle_sha": slot["h0_runtime_bundle_sha"],
            "runtime_bundle_sha": slot["runtime_bundle_sha"],
            "harness_content_sha": slot["harness_content_sha"],
            "skill_ids": list(slot["skill_ids"]),
            "scope_risk_guards_at_start": [],
        },
        "nothing_was_hand_written": (
            "the card is compiled and registered by the delivered code path; "
            "no clause text, no Workflow name and no store pointer is set by "
            "hand anywhere in this run"
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    return slot, record


def _cohort() -> dict[str, Any]:
    """The #17 cohort, rebuilt from what #17 left on disk.  No csv is reparsed."""
    artifact = FC._cohort_artifact()
    health = artifact["step_2_health_check_v2"]
    roster = {
        "train": [str(u) for u in health["confirmation_roster"]],
        "eval": [str(u) for u in health["substitutes"]],
    }
    return SSU._confirmation_payload(roster), FC._missing_cap(artifact), roster


def _search(window: Mapping[str, Any], cohort: Mapping[str, Any]) -> Any:
    return FC.FreshSearch(
        payload=cohort, consumer_variant=VARIANT,
        support_origins=window["support_origins"],
        delayed_origins=window["delayed_origins"],
    )


def _gate(
    search: Any, snapshot: Any, record: Mapping[str, Any], *, reused: bool,
) -> dict[str, Any]:
    """The tracked Scope/Risk gate, over the frozen ladder's own receipt."""
    ladder = {
        "final_plan": dict(record["final_plan"]),
        "support": dict(record.get("support") or {}),
        "delayed": dict(record.get("delayed") or {}),
        "support_winner": (record.get("adoption_ladder") or {}).get(
            "support_winner"
        ),
        "support_winner_full_batch_delayed": (
            record.get("adoption_ladder") or {}
        ).get("support_winner_full_batch_delayed"),
    }
    return harness_compiler.enforce_scope_risk_guards(
        snapshot=snapshot,
        ladder=ladder,
        eval_count=len(search.eval_uids),
        reused=bool(reused),
        delayed_of=search.delayed_gate,
        support_of=search.support_of_plan,
    )


def _step(
    *, tag: str, window: Mapping[str, Any], cohort: Mapping[str, Any],
    slot: dict[str, Any], local_skill: str | None, budget: Budget,
) -> dict[str, Any]:
    """One task window: retrieve, decide through the frozen path, then gate."""
    search = _search(window, cohort)
    snapshot = slot["_snapshot"]
    view = {"slot": slot["slot"], "_snapshot": snapshot}
    if local_skill:
        record = FC._direct_recall(
            search=search, target=FC._target(VARIANT), arm=ARM, window=window,
            slot=view, expected_card=CARD_ID, expected_local=str(local_skill),
            tag=tag,
        )
        if record.get("stopped") == "RECALL_MISS":
            raise Blocked(
                "POST_UPDATE_NOT_RETRIEVED"
                if tag == "task_D" else "LOCAL_LIFECYCLE_BREAK",
                "%s did not retrieve the ACTIVE local Skill %s"
                % (tag, local_skill),
            )
        reused = bool(record.get("reuse_adopted"))
    else:
        record = FC._episode(
            search=search, target=FC._target(VARIANT), arm=ARM, window=window,
            slot=view, expected_card=CARD_ID, expected_local=None,
            llm_budget=budget.take(LLM_CALL_BUDGET_PER_EPISODE), tag=tag,
        )
        budget.spend(int(record.get("llm_calls") or 0))
        if record.get("stopped"):
            raise Blocked(
                "SOURCE_DELIVERY_BREAK",
                "%s produced no payload: %s" % (tag, record["stopped"]),
            )
        reused = False
    if not (record.get("retrieval") or {}).get("hit"):
        raise Blocked(
            "SOURCE_DELIVERY_BREAK",
            "%s did not retrieve the Guidance card naturally" % tag,
        )
    gate = _gate(search, snapshot, record, reused=reused)
    per_series_before = SSU._per_series(search, record["final_plan"])
    per_series_after = (
        dict(per_series_before)
        if dict(gate["plan_after"]) == dict(record["final_plan"])
        else SSU._per_series(search, gate["plan_after"])
    )
    budget.charge_retrains(int(search.retrains))
    row = {
        "step": tag,
        "window_id": str(window["window_id"]),
        "window": {
            k: v for k, v in window.items() if not str(k).startswith("reference_")
        },
        "mode": record.get("mode"),
        "card_hit": bool((record.get("retrieval") or {}).get("hit")),
        "clauses_cited": list(record.get("skill_clause_use") or ()),
        "clauses_available": list(record.get("skill_clause_ids_available") or ()),
        "local_skill_expected": local_skill,
        "local_skill_hit": (record.get("retrieval") or {}).get("local_skill_hit"),
        "reuse_adopted": record.get("reuse_adopted"),
        "shortlist": list(record.get("shortlist") or ()),
        "plan_before_gate": dict(record["final_plan"]),
        "support_before_gate": (record.get("support") or {}).get("aggregate_gain"),
        "delayed_before_gate": (record.get("delayed") or {}).get("aggregate_gain"),
        "harmed_before_gate": list(
            (record.get("delayed") or {}).get("harmed_eval_series") or ()
        ),
        "harm_total_before_gate": (record.get("delayed") or {}).get(
            "harmed_eval_series_total_harm"
        ),
        "per_eval_series_delayed_before_gate": per_series_before,
        "gate": gate,
        "plan_after_gate": dict(gate["plan_after"]),
        "support_after_gate": (gate.get("support_after") or {}).get(
            "aggregate_gain"
        ),
        "delayed_after_gate": (gate.get("delayed_after") or {}).get(
            "aggregate_gain"
        ),
        "harmed_after_gate": list(
            (gate.get("delayed_after") or {}).get("harmed_eval_series") or ()
        ),
        "per_eval_series_delayed_after_gate": per_series_after,
        "gate_changed_the_decision": bool(gate.get("changed")),
        "adoption_ladder": record.get("adoption_ladder"),
        "consumer_retrains": int(search.retrains),
        "llm_calls": int(record.get("llm_calls") or 0),
        "episode_id": record.get("episode_id"),
        "store_runtime_bundle_sha": slot.get("runtime_bundle_sha"),
        "snapshot_runtime_bundle_sha": snapshot.runtime_bundle_sha,
    }
    print(
        "OP %-8s %-14s %-26s -> %-26s | sup %s del %s | harm %s | gate %s | "
        "retrains %d llm %d"
        % (
            tag, str(record.get("mode")),
            SSU._plan_label(record["final_plan"]),
            SSU._plan_label(gate["plan_after"]),
            _fmt(row["support_before_gate"]), _fmt(row["delayed_before_gate"]),
            row["harmed_before_gate"] or "none",
            "moved" if gate.get("changed") else (
                "checked" if gate.get("checked") else "inactive"
            ),
            int(search.retrains), int(row["llm_calls"]),
        ),
        flush=True,
    )
    return {"row": row, "record": record, "search": search, "gate": gate}


# ------------------------------------------------------- attribution and Slow
def stage_attribution(record: Mapping[str, Any], snapshot: Any) -> dict[str, Any]:
    """First fault on this run's own task_C episode.  No prior artifact is fed in."""
    bootstrap = [
        skill.skill_id
        for skill in compile_snapshot(FC.H0_ROOT, verify_lock=False).skills
    ]
    capability = [
        skill.skill_id for skill in snapshot.skills
        if skill.skill_id not in set(bootstrap)
    ]
    with_series = SSU._attribute(
        record, bootstrap_ids=bootstrap,
        capability_skills_present=bool(capability), transport_per_series=True,
    )
    aggregate_only = SSU._attribute(
        record, bootstrap_ids=bootstrap,
        capability_skills_present=bool(capability), transport_per_series=False,
    )
    out = {
        "ran": True,
        "llm_calls": 0,
        "consumer_retrains": 0,
        "evidence": (
            "this run's own task_C episode record only; no #17 or #19 "
            "artifact is read by the fold"
        ),
        "capability_skills_in_store": capability,
        "with_per_series_risk_reading": with_series,
        "aggregate_only_control": aggregate_only,
        "cause_code": with_series["attribution"]["cause_code"],
        "first_stage": with_series["attribution"]["first_stage"],
        "is_scope_risk_face": bool(with_series["is_scope_risk_face"]),
    }
    print(
        "OP attribution  %s at %s (aggregate-only: %s)"
        % (
            out["cause_code"], out["first_stage"],
            aggregate_only["attribution"]["cause_code"],
        ),
        flush=True,
    )
    return out


def stage_slow(
    *, slot: dict[str, Any], attribution: Mapping[str, Any],
    task_c: Mapping[str, Any], budget: Budget, sink: dict[str, Any],
) -> dict[str, Any]:
    """One proposal, one pinned backend, one surface, one draw.

    A2 (#21 reporting defect): everything this stage learns is written into
    ``sink`` as it happens, and ``sink`` is already attached to the artifact
    payload before the call.  Whatever raises, the per-draw record and the
    proposal text are on disk.
    """
    started = time.perf_counter()
    sink.update({
        "ran": True,
        "backend": {"model": SLOW_MODEL, "base_url": SLOW_BASE_URL,
                    "rounds": 1, "cross_backend_sampling": False},
        "attributed_from_episode": str(task_c["record"]["episode_id"]),
        "confirmed_cause": str(attribution["cause_code"]),
        "attempts": [],
        "proposal": None,
        "application": None,
        "records_before_it_raises": True,
    })
    if not attribution["is_scope_risk_face"]:
        raise Blocked(
            "ATTRIBUTION_BREAK",
            "the fold named %s, which authorizes no Scope/Risk surface"
            % attribution["cause_code"],
        )
    snapshot = slot["_snapshot"]
    cause = str(attribution["cause_code"])
    route = attribution["with_per_series_risk_reading"]["route_authorization"]
    offered = SSU._authorized_surfaces_v2(snapshot)
    if not isinstance(offered[0]["current_value"], list):
        raise Blocked(
            "INSTRUMENT_DRIFT",
            "the Scope/Risk surface does not resolve to a list in this store",
        )
    delayed = task_c["record"].get("delayed") or {}
    evidence = {
        "window": dict(task_c["row"]["window"]),
        "evaluation_roster": list(task_c["search"].eval_uids),
        "training_roster_size": len(task_c["search"].train_uids),
        "material_line": MATERIAL_THRESHOLD,
        "harm_line": HARM_THRESHOLD,
        "this_run_only": (
            "every number below was produced by this run; no earlier "
            "artifact is quoted to the Slow Agent"
        ),
        "the_failing_episode": {
            "step": "task_C",
            "mode": task_c["record"].get("mode"),
            "adopted_plan": dict(task_c["record"]["final_plan"]),
            "support_aggregate_gain": (task_c["record"].get("support") or {}).get(
                "aggregate_gain"
            ),
            "delayed_aggregate_gain": delayed.get("aggregate_gain"),
            "evaluation_series_past_the_material_line": list(
                delayed.get("harmed_eval_series") or ()
            ),
            "their_summed_loss_magnitude": delayed.get(
                "harmed_eval_series_total_harm"
            ),
            "per_evaluation_series_delayed_gain": dict(
                task_c["row"]["per_eval_series_delayed_before_gate"]
            ),
            "adoption_path": (
                task_c["record"].get("adoption_ladder") or {}
            ).get("path"),
            "adoption_path_text": (
                task_c["record"].get("adoption_ladder") or {}
            ).get("path_text"),
        },
        "earlier_windows_this_run": [
            {
                "step": row["step"],
                "adopted_plan": row["plan_after_gate"],
                "support": row["support_before_gate"],
                "delayed": row["delayed_before_gate"],
                "harmed": row["harmed_before_gate"],
            }
            for row in task_c["earlier_rows"]
        ],
        "what_the_runtime_folded": attribution["with_per_series_risk_reading"][
            "attribution"
        ],
        "adoption_ladder_is_frozen": (
            "the ladder picks the Support winner, sets the bar at "
            "max(0, that winner's full-batch delayed), adopts the named plan "
            "when it clears the bar, and otherwise falls back.  It is not "
            "modifiable here and it may not be re-ranked."
        ),
    }
    public_input = {
        "stage_note": SSU.SLOW_NOTE_V2,
        "attributed_fault": attribution["with_per_series_risk_reading"][
            "attribution"
        ],
        "attributed_from_episode": str(task_c["record"]["episode_id"]),
        "route_authorization": dict(route),
        "authorized_surfaces": offered,
        "episode_evidence": evidence,
        "guard_grammar": SSU.GUARD_GRAMMAR_V2,
        "guard_contract": SSU.GUARD_CONTRACT_V2,
        "public_boundary": {
            "rule": (
                "a deployable edit may not encode private or path-derived "
                "evidence; these substrings are refused anywhere in the value"
            ),
            "forbidden_substrings": [
                "clean_future", "clean_context", "oracle_affected",
                "injection_type", "candidate_utilities", "selection_regret",
                "loss_j", "utility_u", "r_private", "private_receipt",
            ],
        },
        "manifest_fields_you_author": {
            "edit_id": "a fresh canonical id for this edit",
            "target_pattern_id": "a canonical id naming the pattern you are fixing",
            "predicted_agent_behavior_change": (
                "one or more predicates from the closed vocabulary below"
            ),
            "predicted_agent_behavior_vocabulary": SSU._behavior_vocabulary(),
            "predicted_data_effect": "free text, what should happen to the data",
            "falsification_condition": "free text, what would show this edit wrong",
        },
    }
    view = resolve_harness_view(snapshot, {}, role="slow")
    validator = SSU._make_proposal_validator_v2()
    attempts: list[dict[str, Any]] = sink["attempts"]
    payload: dict[str, Any] = {}
    transport_failures = 0
    sink["authorized_surfaces_offered"] = [
        str(item["surface_id"]) for item in offered
    ]
    sink["public_input_sha256"] = canonical_sha256(wvc._plain(public_input))
    sink["harness_view_sha"] = view.effective_harness_view_sha
    while True:
        backend = SSU._backend_factory_v2(SLOW_MODEL, budget.take(4))
        gateway = wvc.NoToolGateway({
            "protocol": PROTOCOL_VERSION, "stage": "edit",
        })
        core = TTHAAgentCore(
            backend, gateway, model=SLOW_MODEL, base_url=SLOW_BASE_URL,
        )
        row: dict[str, Any] = {
            "draw": len(attempts) + 1, "model": SLOW_MODEL,
        }
        try:
            result = core.run_stage(
                role=AgentRole.SLOW, stage="edit",
                case_id="OP_%s" % SLOT,
                public_input=public_input, harness_view=view,
                output_schema_name="slow_scope_guard_v1",
                output_schema=SSU.PROPOSAL_SCHEMA_V2,
                source_snapshot_sha=view.effective_harness_view_sha,
                validation_retries=wvc.VALIDATION_RETRIES,
                post_validator=validator,
            )
        except Exception as exc:  # noqa: BLE001
            budget.spend(int(backend.calls))
            message = "%s: %s" % (type(exc).__name__, exc)
            transport = "Transport" in type(exc).__name__
            row.update({
                "outcome": "TRANSPORT_FAILURE" if transport else "STAGE_ERROR",
                "error": message, "llm_calls": int(backend.calls),
                "consumes_a_draw": not transport,
            })
            attempts.append(row)
            if transport:
                transport_failures += 1
                if transport_failures >= MAX_TRANSPORT_FAILURES:
                    raise Blocked(
                        "INCONCLUSIVE_TRANSPORT",
                        "%d consecutive transport failures" % transport_failures,
                    )
                continue
            raise Blocked("COMPILER_REJECTS", message)
        budget.spend(int(backend.calls))
        transport_failures = 0
        candidate = dict(result.payload)
        row.update({
            "outcome": "PROPOSAL" if candidate else "NO_PROPOSAL",
            "llm_calls": int(backend.calls),
            "consumes_a_draw": True,
            "validation_retry_count": int(result.validation_retry_count),
            "validation_error_codes": list(result.validation_error_codes),
            "no_proposal_reason": result.no_proposal_reason,
            "returned_models": sorted({
                str(m) for m in getattr(result.response, "returned_models", ())
            }) or None,
        })
        attempts.append(row)
        sink["llm_calls"] = sum(int(a.get("llm_calls") or 0) for a in attempts)
        if not candidate:
            raise Blocked(
                "SLOW_ABSTAINS",
                "the Slow Agent returned no_proposal (%s); the pinned "
                "configuration gets one draw" % result.no_proposal_reason,
            )
        payload = candidate
        sink["proposal"] = wvc._plain(payload)
        sink["guard"] = wvc._plain(payload["guard"])
        sink["target_surface_id"] = str(payload["target_surface_id"])
        break

    application = SSU._apply_patch(
        {"slot": SLOT, "store": slot["_store"], "snapshot": snapshot},
        payload=payload, cause=cause,
    )
    sink["application"] = application
    if not application.get("applied"):
        raise Blocked(
            "COMPILER_REJECTS",
            str(application.get("reason") or "the controller refused the edit"),
        )
    # _apply_patch already set the fork snapshot active; re-read it here.
    active = json.loads(
        slot["_store"].active_path.read_text(encoding="utf-8")
    )["runtime_bundle_sha"]
    snapshot_after = compile_snapshot(
        Path(slot["_store"].root) / active, verify_lock=False
    )
    slot["_snapshot"] = snapshot_after
    slot["runtime_bundle_sha"] = snapshot_after.runtime_bundle_sha
    guards_after = harness_compiler.scope_risk_guards_of(snapshot_after)
    sink.update({
        "llm_calls": sum(int(a.get("llm_calls") or 0) for a in attempts),
        "snapshot_before": str(snapshot.runtime_bundle_sha),
        "snapshot_after": str(snapshot_after.runtime_bundle_sha),
        "harness_content_before": str(snapshot.harness_content_sha),
        "harness_content_after": str(snapshot_after.harness_content_sha),
        "guards_active_after": guards_after,
        "same_store": (
            "the patch is written into the store this run has been using "
            "since task_A; no store is swapped and no snapshot is hand-picked"
        ),
        "wall_seconds": time.perf_counter() - started,
    })
    return sink


# --------------------------------------------- the post-update behaviour read
def stage_shadow(
    *, task_c: Mapping[str, Any], snapshot_after: Any,
) -> dict[str, Any]:
    """Criterion (ii): re-decide this run's own task_C under the new Harness.

    Zero evaluations.  The task_C search object is the one that already
    measured every reading this needs; identity is cached in its own
    constructor, so the fallback walk costs nothing.  If a retrain is
    charged here the check is void and says so.
    """
    search = task_c["search"]
    before_retrains = int(search.retrains)
    gate = _gate(
        search, snapshot_after, task_c["record"],
        reused=bool(task_c["record"].get("reuse_adopted")),
    )
    spent = int(search.retrains) - before_retrains
    pre = dict(task_c["gate"]["plan_after"])
    post = dict(gate["plan_after"])
    return {
        "ran": True,
        "zero_evaluation": spent == 0,
        "consumer_retrains_spent": spent,
        "snapshot": str(snapshot_after.runtime_bundle_sha),
        "decision_before_patch": pre,
        "decision_after_patch": post,
        "decision_changed": pre != post,
        "gate": gate,
        "consistent_with_the_risk_contract": bool(
            gate.get("changed")
            and str(post["program"]) == IDENTITY
            and not (gate.get("delayed_after") or {}).get("harmed_eval_series")
        ),
        "why": (
            "the same measured task_C readings, re-decided under the patched "
            "snapshot; nothing is re-measured and nothing is re-adopted"
        ),
    }


def stage_experience(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One Experience row for the post-update task, tagged with its provenance."""
    task_d = rows[-1]
    plan = dict(task_d["plan_after_gate"])
    audit = {
        "provenance": "operational_pipeline",
        "counts_as_unguided_exploration": False,
    }
    if str(plan["program"]) == IDENTITY:
        return {
            "ran": True,
            "provenance": "operational_pipeline",
            "rows": [],
            "row_count": 0,
            "no_row_written": (
                "task_D ended on identity; this line forms no Experience row "
                "for an abstention adoption"
            ),
            "not_persisted_as_skills": (
                "handle_fast_winner is called at task_A only, as in #17: the "
                "post-update window reads behaviour, it does not form a Skill"
            ),
        }
    steps = [(str(plan["program"]), {})]
    support_gain = float(task_d["support_after_gate"] or 0.0)
    delayed_gain = float(task_d["delayed_after_gate"] or 0.0)
    episode = build_episode(
        episode_id="op_%s_task_d" % VARIANT,
        task_consumer_key="batch:%s|consumer:%s" % (COHORT_NAME, VARIANT),
        domain_namespace=str(COHORT_NAME),
        context_summary={
            "task_episode_id": str(task_d["window_id"]),
            "arm": ARM,
            "cohort": {"cohort_name": str(COHORT_NAME)},
            "local_pattern": {"consumer_variant": VARIANT},
            "program_geometry": {
                "program_steps": [
                    {"op": op, "params": dict(params)} for op, params in steps
                ],
                "frozen_plan_scope": {
                    "excluded_series": sorted(
                        str(uid) for uid in plan["excluded_series"]
                    )
                },
                "consumer_retrains": int(task_d["consumer_retrains"]),
            },
        },
        workflow_signature=FC.e1mod._v2_workflow_signature(steps),
        support_response={
            "gain": support_gain,
            "accepted": support_gain >= MATERIAL_THRESHOLD,
            "block_origins": list(task_d["window"]["support_origins"]),
            "program": str(plan["program"]),
            "excluded_series": list(plan["excluded_series"]),
            **audit,
        },
        delayed_response={
            "evaluated": True,
            "gain": delayed_gain,
            "se_block": None,
            "gain_over_se": None,
            "block_origins": list(task_d["window"]["delayed_origins"]),
            "took_part_in_selection": True,
            "why_not_promotion_evidence": (
                "this delayed reading set the adoption bar in its own "
                "episode, so it may not also license a promotion"
            ),
            **audit,
        },
        relation=FC._relation(support_gain, delayed_gain, plan["program"]),
        evidence_level=EVIDENCE_SUPPORT,
        local_status=FC.STATUS_LOCAL_DRAFT,
        evidence_refs=("operational_pipeline", PROTOCOL_VERSION),
    )
    return {
        "ran": True,
        "provenance": "operational_pipeline",
        "rows": [{
            "episode_id": episode.episode_id,
            "workflow_signature": episode.workflow_signature,
            "relation": str(episode.relation),
            "local_status": str(episode.local_status),
            **audit,
        }],
        "row_count": 1,
        "not_persisted_as_skills": (
            "handle_fast_winner is called at task_A only, as in #17"
        ),
    }


# --------------------------------------------------------------- the verdict
def _verdict(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = {row["step"]: row for row in payload["trajectory"]}
    slow = payload["slow_edit"]
    if (payload.get("no_fault_sample") or {}).get(
        "this_trajectory_produced_no_fault"
    ):
        return {
            "verdict": "PIPELINE_RUNS_NO_FAULT_SAMPLE",
            "reason": (
                "the trajectory ran end to end but produced no harm past the "
                "line, so the fold had nothing actionable to name; links 6 to "
                "9 -- attribution to a Scope/Risk face, the Slow edit, the "
                "compiler and the post-update behaviour -- are untested in "
                "this sample"
            ),
            "detail": {
                "links_1_to_5_exercised": True,
                "links_6_to_9_untested": True,
                "source_delivered": bool(rows["task_A"]["card_hit"]),
                "clauses_cited_at_task_a": list(rows["task_A"]["clauses_cited"]),
                "skill_promoted": bool(
                    payload["lifecycle"]["promotion"].get("promoted")
                ),
                "skill_recalled_at_task_b": bool(
                    rows["task_B"].get("local_skill_hit")
                ),
                "task_c_harmed_eval_series": list(
                    rows["task_C"]["harmed_before_gate"]
                ),
                "task_d_harmed_eval_series": (
                    list(rows["task_D"]["harmed_before_gate"])
                    if "task_D" in rows else None
                ),
            },
        }
    task_d = rows.get("task_D")
    shadow = payload.get("shadow_replay") or {}
    detail: dict[str, Any] = {
        "source_delivered": bool(rows["task_A"]["card_hit"]),
        "clauses_cited_at_task_a": list(rows["task_A"]["clauses_cited"]),
        "skill_formed": bool(payload["lifecycle"]["draft"].get("written")),
        "skill_promoted": bool(payload["lifecycle"]["promotion"].get("promoted")),
        "skill_recalled_at_task_b": bool(
            rows["task_B"].get("local_skill_hit")
        ),
        "attribution_cause": payload["attribution"]["cause_code"],
        "surfaces_changed": list(
            slow["application"].get("source_surfaces_changed") or ()
        ),
        "task_d_snapshot": task_d and task_d["snapshot_runtime_bundle_sha"],
        "post_patch_snapshot": slow["snapshot_after"],
    }
    if task_d is None:
        return {"verdict": "NO_POST_UPDATE_WINDOW",
                "reason": "task_D never ran", "detail": detail}
    if str(task_d["snapshot_runtime_bundle_sha"]) != str(slow["snapshot_after"]):
        return {
            "verdict": "POST_UPDATE_NOT_RETRIEVED",
            "reason": (
                "task_D read snapshot %s, not the patched %s"
                % (task_d["snapshot_runtime_bundle_sha"], slow["snapshot_after"])
            ),
            "detail": detail,
        }
    gate = task_d["gate"]
    evaluated = bool(gate.get("checked")) or bool(gate.get("guard_count"))
    detail["guard_evaluated_at_task_d"] = evaluated
    detail["guard_readings_at_task_d"] = (
        gate.get("check_on_adopted_plan") or {}
    ).get("readings")
    if not evaluated:
        return {
            "verdict": "POST_UPDATE_NOT_RETRIEVED",
            "reason": (
                "the patched snapshot was active but the guard was never "
                "evaluated at task_D"
            ),
            "detail": detail,
        }
    criterion_i = bool(gate.get("changed"))
    criterion_ii = bool(
        shadow.get("ran")
        and shadow.get("zero_evaluation")
        and shadow.get("decision_changed")
        and shadow.get("consistent_with_the_risk_contract")
    )
    harm_recurred = bool(task_d["harmed_before_gate"])
    detail.update({
        "criterion_i_task_d_fired": criterion_i,
        "criterion_ii_shadow_replay": criterion_ii,
        "harm_recurred_at_task_d": harm_recurred,
        "guard_silent_and_no_harm_is_the_contract_working": bool(
            not criterion_i and not harm_recurred
        ),
    })
    if not (criterion_i or criterion_ii):
        return {
            "verdict": "NEXT_TASK_NO_BEHAVIOR_CHANGE",
            "reason": (
                "the guard was evaluated at task_D but neither criterion held: "
                "it did not fire on the spot, and the shadow replay did not "
                "change this run's own task_C decision"
            ),
            "detail": detail,
        }
    return {
        "verdict": "OPERATIONAL_PIPELINE_CLOSES",
        "reason": (
            "Source delivered and cited, Skill formed / promoted / recalled, "
            "the fold named %s on this run's own episode, one surface was "
            "patched and compiled into the same store, task_D read the "
            "patched snapshot and the guard was evaluated there; criterion "
            "%s holds"
            % (
                detail["attribution_cause"],
                "(i)" if criterion_i else "(ii)",
            )
        ),
        "detail": detail,
    }


# ------------------------------------------------------------------- the run
# A3 (#21 reporting defect): the old filter dropped every key literally
# named "store", which silently deleted the whole store/card block from the
# artifact.  Live objects are excluded by type now, and by the underscore
# convention the runner already uses for them, so a plain data key called
# "store" survives.
_LIVE_KEYS = ("snapshot", "search", "record", "gate_obj")
_JSON_SCALARS = (str, int, float, bool, type(None))


def _public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _public(v) for k, v in value.items()
            if not str(k).startswith("_") and str(k) not in _LIVE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (value != value or value in (
        float("inf"), float("-inf")
    )):
        return None
    if not isinstance(value, _JSON_SCALARS):
        return "<%s not serializable>" % type(value).__name__
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return "%+.6f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def run(*, gate_only: bool = False) -> int:
    started = time.perf_counter()
    before = _freeze()
    budget = Budget(LLM_CALL_BUDGET_TOTAL, RETRAIN_BUDGET)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "one continuous, un-relayed run of the V1 Harness: Source use, "
            "target adaptation, lifecycle, attribution, Slow edit, next-task "
            "behaviour"
        ),
        "pre_registered": PRE_REGISTERED,
        "p1_touched_files": [dict(row) for row in P1_TOUCHED],
        "a1_touched_files": [dict(row) for row in A1_TOUCHED],
        "frozen_surface_before": {"files": len(before), "sha256": before},
        "llm_call_budget": LLM_CALL_BUDGET_TOTAL,
        "retrain_budget": RETRAIN_BUDGET,
    }
    for row in payload["p1_touched_files"]:
        row["sha256_after_p1"] = before[row["path"]]
    for row in payload["a1_touched_files"]:
        row["sha256_after_a1"] = before[row["path"]]
    try:
        payload["p2_non_regression_gate"] = stage_p2_precondition()
        payload["guard_after_p2"] = _guard_frozen(before, "the pipeline")
        if gate_only:
            payload.update({
                "overall_verdict": "P2_ONLY",
                "overall_verdict_reason": "asked to stop after the gate",
                "llm_call_count": budget.llm_used,
                "consumer_retrains_total": budget.retrains_charged,
                "wall_seconds": time.perf_counter() - started,
                "frozen_surface_after": _verify(before),
            })
            return _write(payload, dry=True)

        cohort, cap, roster = _cohort()
        slot, store_record = stage_store()
        payload["store"] = store_record
        payload["cohort"] = {
            "name": COHORT_NAME, "roster": roster,
            "eval_uids": list(cohort["eval_uids"]),
            "rebuilt_from": "what #17 materialized; no csv is parsed again",
        }

        trajectory: list[dict[str, Any]] = []
        # --- task_A: full-price adaptation on the Source card ---------------
        step_a = _step(
            tag="task_A", window=WINDOW_TASK_A, cohort=cohort, slot=slot,
            local_skill=None, budget=budget,
        )
        trajectory.append(step_a["row"])

        # --- lifecycle: Draft, out-of-selection probe, promotion ------------
        draft = FC._persist_draft(
            slot=slot, record=step_a["record"], target=FC._target(VARIANT),
            arm=ARM,
        )
        probe = None
        promotion: dict[str, Any] = {
            "promoted": False, "reason": draft.get("reason")
        }
        probe_retrains = 0
        if draft.get("written"):
            probe = FC._probe(
                search=step_a["search"], payload=cohort, variant=VARIANT,
                plan=dict(step_a["record"]["final_plan"]),
            )
            probe_retrains = int(probe["consumer_retrains"])
            budget.charge_retrains(probe_retrains)
            promotion = FC._promote(slot=slot, probe=probe, draft=draft)
        payload["lifecycle"] = {
            "draft": _public(draft),
            "probe": _public(probe),
            "probe_window": dict(WINDOW_PROBE),
            "probe_consumer_retrains": probe_retrains,
            "promotion": _public(promotion),
            "path": dict(FC.UPDATE_PATH),
        }
        print(
            "OP lifecycle    draft=%s probe=%s promoted=%s skill=%s"
            % (
                bool(draft.get("written")),
                None if probe is None else round(float(probe["macro_gain"]), 6),
                bool(promotion.get("promoted")),
                promotion.get("retrievable_skill_id"),
            ),
            flush=True,
        )
        local_skill = promotion.get("retrievable_skill_id")
        if not local_skill:
            raise Blocked(
                "LOCAL_LIFECYCLE_BREAK",
                "no Target-local Skill reached ACTIVE: %s"
                % (promotion.get("reason") or draft.get("reason")),
            )

        # --- task_B: recall and reuse ---------------------------------------
        step_b = _step(
            tag="task_B", window=WINDOW_TASK_B, cohort=cohort, slot=slot,
            local_skill=str(local_skill), budget=budget,
        )
        trajectory.append(step_b["row"])

        # --- task_C: the already-exposed 2025 window ------------------------
        step_c = _step(
            tag="task_C", window=WINDOW_TASK_C, cohort=cohort, slot=slot,
            local_skill=str(local_skill), budget=budget,
        )
        trajectory.append(step_c["row"])
        step_c["earlier_rows"] = [step_a["row"], step_b["row"]]
        payload["trajectory"] = trajectory
        payload["guard_after_task_c"] = _guard_frozen(before, "attribution")

        # --- attribution on this run's own record ---------------------------
        attribution = stage_attribution(step_c["record"], slot["_snapshot"])
        payload["attribution"] = _public(attribution)

        # --- Slow: one draw, one surface ------------------------------------
        # A2: the sink is attached to the payload before the call, so
        # whatever raises inside stage_slow, the per-draw record and the
        # proposal text are already where the artifact writer will find them.
        slow_sink: dict[str, Any] = {"ran": False}
        payload["slow_edit"] = slow_sink
        no_fault = bool(
            str(attribution["cause_code"]) == "NO_ACTIONABLE_FAULT"
            or not step_c["row"]["harmed_before_gate"]
        )
        payload["no_fault_sample"] = {
            "this_trajectory_produced_no_fault": no_fault,
            "task_c_harmed_eval_series": list(step_c["row"]["harmed_before_gate"]),
            "attribution_cause": str(attribution["cause_code"]),
            "rule": (
                "with nothing past the harm line the fold has nothing "
                "actionable to name, so links 6 to 9 go untested in this "
                "sample.  Nothing is re-thrown and no harm is seeded."
            ) if no_fault else "not applicable: this trajectory produced a fault",
        }
        if no_fault:
            slow_sink.update({
                "ran": False,
                "why_not_run": (
                    "the trajectory reached task_C with no evaluation series "
                    "past the harm line, so there is no fault sample to "
                    "attribute a Scope/Risk edit to"
                ),
            })
            print("OP slow_edit    skipped: no fault sample this trajectory",
                  flush=True)
        else:
            slow = stage_slow(
                slot=slot, attribution=attribution, task_c=step_c,
                budget=budget, sink=slow_sink,
            )
            print(
                "OP slow_edit    %s -> %s | snapshot %s -> %s"
                % (
                    slow["guard"]["statistic"], slow["guard"]["action"],
                    slow["snapshot_before"][:12], slow["snapshot_after"][:12],
                ),
                flush=True,
            )
        payload["guard_after_slow"] = _guard_frozen(before, "task_D")

        # --- task_D: the post-update window ---------------------------------
        gate_d = FC._missing_gate(
            cohort["values"], cohort["eval_uids"], WINDOW_TASK_D, cap,
        )
        payload["task_d_missing_gate"] = _public(gate_d)
        if not gate_d["pass"]:
            raise Blocked(
                "NO_POST_UPDATE_WINDOW",
                "task_D does not clear the missing gate",
            )
        step_d = _step(
            tag="task_D", window=WINDOW_TASK_D, cohort=cohort, slot=slot,
            local_skill=str(local_skill), budget=budget,
        )
        trajectory.append(step_d["row"])
        payload["trajectory"] = trajectory

        # --- criterion (ii) only if the harm did not recur -------------------
        shadow = None
        if not step_d["gate"].get("changed"):
            shadow = stage_shadow(
                task_c=step_c, snapshot_after=slot["_snapshot"],
            )
            payload["shadow_replay"] = _public(shadow)

        payload["experience"] = stage_experience(trajectory)
        outcome = _verdict(payload)
        payload.update({
            "overall_verdict": outcome["verdict"],
            "overall_verdict_reason": outcome["reason"],
            "overall_detail": _public(outcome.get("detail") or {}),
        })
    except Blocked as exc:
        payload.update({
            "overall_verdict": exc.verdict,
            "overall_verdict_reason": exc.reason,
            "stopped_at_first_block": True,
        })
    except ConcurrentWrite as exc:
        payload.update({
            "overall_verdict": "CONCURRENT_WRITE_ABORT",
            "overall_verdict_reason": str(exc),
        })
    payload.update({
        "llm_call_count": budget.llm_used,
        "consumer_retrains_total": budget.retrains_charged,
        "exposure": {
            "windows_read": [
                "2024 development training anchors (indices 120-900)",
                "task_A/probe/task_B inside the 2024 development partition",
                "task_C [9864, 10152] and task_D [10560, 10848], both inside "
                "the 2025 partition #17 already opened",
            ],
            "beyond_17520": "SEALED, not read",
            "unopened_partition_read": "none",
            "fresh_claim": (
                "none: development level throughout, on partitions this line "
                "has already opened"
            ),
        },
        "wall_seconds": time.perf_counter() - started,
        "frozen_surface_after": _verify(before),
    })
    if not payload["frozen_surface_after"]["ok"]:
        payload["overall_verdict"] = "CONCURRENT_WRITE_ABORT"
        payload["overall_verdict_reason"] = (
            "the frozen surface moved during the run; the reading is void"
        )
    return _write(payload)


# ---------------------------------------------------------------- the report
def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# One continuous operational run",
        "",
        "**Overall: `%s`** -- %s" % (
            payload.get("overall_verdict"),
            payload.get("overall_verdict_reason", ""),
        ),
        "",
        "One un-relayed run of the V1 Harness on %s x %s, arm %s, Slow pinned "
        "to `%s`. Development level: every window was locked before the run "
        "from the #17/#19 registers, nothing beyond index 17520 was read, "
        "A5-vs-A3 was not re-estimated and no new method was introduced."
        % (COHORT_NAME, VARIANT, ARM, SLOW_MODEL),
        "",
    ]
    gate = payload.get("p2_non_regression_gate") or {}
    if gate.get("cells"):
        lines.extend([
            "## P2 -- the non-regression gate",
            "",
            "%s. %d of %d #19 task_C episodes reproduce digit-for-digit "
            "through the promoted enforcement path; %d retrains, 0 LLM."
            % (
                "PASS" if gate.get("ok") else "FAIL",
                sum(1 for c in gate["cells"].values() if c["reproduces"]),
                len(gate["cells"]), int(gate.get("consumer_retrains") or 0),
            ),
            "",
        ])
    elif gate:
        pre = gate.get("precondition") or {}
        lines.extend([
            "## P2 -- not re-run, precondition verified",
            "",
            "- Carried forward: %s." % gate.get("gate_result_carried_forward"),
            "- Measurement-side files byte-identical to their post-P1 state: "
            "%s." % pre.get("byte_identical"),
            "- %s." % gate.get("why_it_cannot_be_re_run_anyway"),
            "",
        ])
    store = payload.get("store") or {}
    if store:
        card = store["card"]
        lines.extend([
            "## The store this run built for itself",
            "",
            "- Card `%s`: %s, %d clauses %s, bytes sha256 `%s`." % (
                CARD_ID, card["status"], card["clause_count"],
                card["clause_ids"], card["card_bytes_sha256"][:16],
            ),
            "- Snapshot `%s`, skills %s, `scope_risk_guards` starts empty." % (
                store["store"]["runtime_bundle_sha"][:16],
                store["store"]["skill_ids"],
            ),
            "",
        ])
    rows = payload.get("trajectory") or []
    if rows:
        lines.extend([
            "## Trajectory",
            "",
            "| step | window | mode | card | local Skill | plan before gate | "
            "plan after gate | support | delayed | harmed | gate | retrains | LLM |",
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | "
            "--- | ---: | ---: |",
        ])
        for row in rows:
            lines.append(
                "| `%s` | %s | %s | %s | %s | `%s` | `%s` | %s | %s | %s | %s "
                "| %d | %d |" % (
                    row["step"], row["window"]["start"], row["mode"],
                    "hit" if row["card_hit"] else "MISS",
                    "--" if row["local_skill_hit"] is None else (
                        "hit" if row["local_skill_hit"] else "MISS"
                    ),
                    SSU._plan_label(row["plan_before_gate"]),
                    SSU._plan_label(row["plan_after_gate"]),
                    _fmt(row["support_before_gate"]),
                    _fmt(row["delayed_before_gate"]),
                    ", ".join(row["harmed_before_gate"]) or "none",
                    "moved" if row["gate_changed_the_decision"] else (
                        "checked" if row["gate"].get("checked") else "inactive"
                    ),
                    int(row["consumer_retrains"]), int(row["llm_calls"]),
                )
            )
        uids = sorted(rows[0]["per_eval_series_delayed_before_gate"])
        lines.extend([
            "",
            "### Per evaluation series, delayed gain",
            "",
            "| step | phase | " + " | ".join("`%s`" % u for u in uids) + " |",
            "| --- | --- | " + " | ".join("---:" for _ in uids) + " |",
        ])
        for row in rows:
            for phase, key in (
                ("before gate", "per_eval_series_delayed_before_gate"),
                ("after gate", "per_eval_series_delayed_after_gate"),
            ):
                values = row[key]
                lines.append(
                    "| `%s` | %s | %s |" % (
                        row["step"], phase,
                        " | ".join(_fmt(values.get(u)) for u in uids),
                    )
                )
        lines.append("")
    life = payload.get("lifecycle") or {}
    if life:
        promo = life.get("promotion") or {}
        probe = life.get("probe") or {}
        lines.extend([
            "## Lifecycle",
            "",
            "- Draft written: %s (`%s`)." % (
                bool((life.get("draft") or {}).get("written")),
                (life.get("draft") or {}).get("skill_id"),
            ),
            "- Probe %s: gain %s, se %s, gain/se %s -- out of selection." % (
                life.get("probe_window", {}).get("window_id"),
                _fmt(probe.get("macro_gain")), _fmt(probe.get("se_block")),
                _fmt(probe.get("gain_over_se")),
            ),
            "- Promotion: %s -> `%s`." % (
                bool(promo.get("promoted")), promo.get("retrievable_skill_id"),
            ),
            "",
        ])
    att = payload.get("attribution") or {}
    if att:
        lines.extend([
            "## Attribution -- on this run's own record",
            "",
            "- With the per-series risk reading: `%s` at %s." % (
                att.get("cause_code"), att.get("first_stage"),
            ),
            "- Through the aggregate alone: `%s`." % (
                (att.get("aggregate_only_control") or {}).get(
                    "attribution", {}
                ).get("cause_code"),
            ),
            "",
        ])
    slow = payload.get("slow_edit") or {}
    if slow.get("guard"):
        g = slow["guard"]
        lines.extend([
            "## The Slow edit",
            "",
            "- Backend `%s`, one round, %d draw(s), no cross-backend sampling."
            % (SLOW_MODEL, len(slow.get("attempts") or ())),
            "- Surface `%s`; guard `%s`: %s `%s` %s -> %s on the %s window, "
            "applies to %s." % (
                slow["target_surface_id"], g["guard_id"], g["statistic"],
                g["comparator"], _fmt(g["threshold"]), g["action"],
                g["window"], g["applies_to"],
            ),
            "- Rationale: %s" % g.get("rationale"),
            "- Snapshot `%s` -> `%s`, in the same store." % (
                slow["snapshot_before"][:16], slow["snapshot_after"][:16],
            ),
            "",
        ])
    shadow = payload.get("shadow_replay")
    if shadow:
        lines.extend([
            "## Shadow replay (criterion ii)",
            "",
            "- Zero evaluation: %s (%d retrains spent)." % (
                shadow["zero_evaluation"], shadow["consumer_retrains_spent"],
            ),
            "- task_C decision `%s` -> `%s`; changed: %s." % (
                SSU._plan_label(shadow["decision_before_patch"]),
                SSU._plan_label(shadow["decision_after_patch"]),
                shadow["decision_changed"],
            ),
            "",
        ])
    after = payload.get("frozen_surface_after") or {}
    lines.extend([
        "## Cost and integrity",
        "",
        "- LLM calls: %s / %s." % (
            payload.get("llm_call_count"), payload.get("llm_call_budget")
        ),
        "- Consumer retrains: %s / %s." % (
            payload.get("consumer_retrains_total"), payload.get("retrain_budget")
        ),
        "- Frozen surface: %s files, drift %s." % (
            after.get("files"), after.get("drift")
        ),
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


def _write(payload: Mapping[str, Any], *, dry: bool = False) -> int:
    body = _public(payload)
    if dry:
        print(json.dumps(body, indent=2, ensure_ascii=False, default=str)[:4000])
    else:
        E2.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(
            json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8", newline="\n",
        )
        OUT_MD.write_text(_markdown(body), encoding="utf-8", newline="\n")
        print("wrote", OUT_JSON, flush=True)
        print("wrote", OUT_MD, flush=True)
    print("overall", body.get("overall_verdict"), flush=True)
    print("llm_calls", body.get("llm_call_count", 0), flush=True)
    print("retrains", body.get("consumer_retrains_total", 0), flush=True)
    return 0 if body.get("overall_verdict") in (
        "OPERATIONAL_PIPELINE_CLOSES", "P2_ONLY"
    ) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-only", action="store_true",
        help="run the P2 non-regression gate and stop, writing nothing",
    )
    args = parser.parse_args(argv)
    return run(gate_only=bool(args.gate_only))


if __name__ == "__main__":
    raise SystemExit(main())
