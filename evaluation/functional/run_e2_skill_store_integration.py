"""Does the compiled recipe Skill survive the real channel?

The bridge run showed that cross-cohort recipe evidence, compiled
deterministically into a text card, changes what Fast proposes and in the
corrected-instrument reading moves all three targets the right way.  But the
card reached the Agent because the runner pasted it into the prompt.  Nothing
was registered anywhere, nothing was retrieved, and the adopted plan formed no
Target-local lifecycle record.

This slice keeps the signal and the targets fixed and changes only the
channel.  The three leave-one-cohort-out cards are registered into the
existing Skill store under the store's own ``skill-entry/1`` schema, the Fast
stage receives whatever ``resolve_harness_view`` returns for the Task Context,
and the plan it adopts is written through the existing Experience lifecycle.
Both arms send byte-identical public input: the only difference between A5 and
A3 is what is in the store they resolve against.

It is not new evidence that the signal transfers -- these are the same three
targets the bridge already measured.  It measures channel loss.

**Engineering integration measurement, not authorization evidence.**  The
registered card carries no executable program and no tool: it is retrievable
guidance, and adoption still has to pass this batch's own Support evaluation
and the delayed gate.  No TRY right is granted, no Skill is promoted, and no
Slow path runs.

Writes ``artifacts/functional/e2/skill_store_integration_v1.json`` and
``.md``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from collections import Counter
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
import run_e2_recipe_experience_to_skill as bridge  # noqa: E402
import run_e2_warm_vs_cold_recipe_search as wvc  # noqa: E402

from evaluation.functional.task_episode_harness.agentic.runner import (  # noqa: E402
    _default_backend_factory,
)
from evaluation.functional.task_episode_harness.normal_flow import (  # noqa: E402
    NF_BASE_URL,
    NF_MODEL,
)
from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    load_learned_skill_entry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    EVIDENCE_SUPPORT,
    RELATION_ABSTAIN,
    RELATION_CONFLICT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_DRAFT,
    build_episode,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)

PROTOCOL_VERSION = "skill_store_integration_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "skill_store_integration_v1.json"
OUT_MD = E2 / "skill_store_integration_v1.md"
CARDS_JSON = E2 / "recipe_skill_cards_v1.json"
BRIDGE_JSON = E2 / "recipe_skill_bridge_v1.json"
REPLAY_JSON = E2 / "recipe_skill_bridge_v2_replay.json"

# A namespace of its own.  Nothing here is written into methods/ttha/harness/h0
# and nothing is written into any store an earlier line already uses.
NAMESPACE = "recipe_batch_guidance_v1"
STORE_ROOT = PROJECT_ROOT / "_scratch" / "skill_store" / NAMESPACE
H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"

TARGETS = bridge.TARGETS
TREATMENTS = bridge.TREATMENTS
IDENTITY = bridge.IDENTITY
EVALUATION_BUDGET = bridge.EVALUATION_BUDGET
MATERIAL_THRESHOLD = float(bch.MATERIAL_THRESHOLD)
QUALITY_DELTA_THRESHOLD = bridge.QUALITY_DELTA_THRESHOLD
LLM_CALL_BUDGET_TOTAL = 40
LLM_CALL_BUDGET_PER_ARM_TARGET = 5
VALIDATION_RETRIES = wvc.VALIDATION_RETRIES
EXPERIENCE_PROVENANCE = "skill_store_integration"
E2_DOMAIN = "batch_recipe_e2"
TASK_CONSUMER_KEY = "forecast|ridge|sMASE"

SKILL_ID = {
    target_id: "recipe_batch_guidance_%s_v1" % target_id.lower()
    for target_id in TARGETS
}
ARM_ORDER: tuple[tuple[str, str], ...] = tuple(
    (target_id, arm)
    for target_id in ("T1", "T2", "T3")
    for arm in ("A3", "A5")
)


# --------------------------------------------------------- provenance policy
# Written before registration and reproduced verbatim in the report.  This is
# the standing claim about what these cards are and what they may do.
PROVENANCE_POLICY: dict[str, Any] = {
    "source_class": "deterministic_recipe_compilation",
    "why_this_class": [
        "the whole program menu was enumerated on every source cell, so no "
        "program reached the card by being the one somebody happened to try",
        "no proposal step and no model chose which records to keep: the "
        "compiler is a fixed rule over committed rows, so there is no "
        "proposal-selection bias to inherit",
        "every row it reads was measured on already-exposed development "
        "origins inside the sealed boundary, and the compiler reads committed "
        "artifacts only -- it never touches data",
    ],
    "authorization_scope": "GUIDANCE",
    "what_guidance_means": (
        "the card is retrievable knowledge for the proposal stage.  It carries "
        "no frozen program and no allowed tool, so it can never enter the "
        "candidate pool on its own; a plan it suggests is adopted only after "
        "this batch's own Support evaluation and the delayed gate, exactly as "
        "an unguided plan would be"
    ),
    "what_is_not_granted": (
        "no confirmation-free TRY right, no execution right, no promotion to "
        "an active Skill, and no standing beyond the store namespace it was "
        "registered into"
    ),
    "carrier_guards_asserted_at_registration": [
        "allowed_tools is empty",
        "the body carries no 'Frozen program steps:' marker, so "
        "_skill_frozen_candidates yields nothing from it",
        "risk_guards records advises_the_proposal_stage_only, "
        "never_supplies_a_candidate and requires_target_support",
    ],
    "loco": (
        "each card was compiled with every record measured on its own target "
        "cohort withheld, and each is registered into a store namespace of its "
        "own so that a target can never resolve another target's card"
    ),
}


PRE_REGISTERED: dict[str, Any] = {
    "fixed_before_the_first_llm_call": True,
    "what_this_slice_changes": (
        "only the channel.  Same three targets, same windows, same compiled "
        "cards, same corrected adoption ladder.  It is not new evidence that "
        "the signal transfers"
    ),
    "registration": (
        "the three cards are loaded through the store's own "
        "load_learned_skill_entry and written into skills/learned of a fork of "
        "a materialized h0; no store, retrieval or lifecycle code is modified. "
        "A card the loader rejects makes that target SCHEMA_BLOCKED and the "
        "interface that refused it is reported"
    ),
    "arms": {
        "A3": "resolves against a store carrying no learned card",
        "A5": "resolves against a store carrying exactly this target's card",
        "parity": (
            "the public input is byte-identical between the arms; the store "
            "trees differ only in skills/learned and the two files whose "
            "content sha depends on it"
        ),
    },
    "retrieval": (
        "the card must be returned by resolve_harness_view for this Task "
        "Context.  The runner never places the card in the prompt: what Fast "
        "reads is the resolved Harness the core renders from the view.  A "
        "target whose card is not returned is RETRIEVAL_MISS and is not "
        "rescued"
    ),
    "instrument": (
        "the corrected v2 ladder: the Support winner is the highest-Support "
        "evaluated full-batch plan and is eligible only when its Support is "
        "positive; bar = max(0, that plan's full-batch delayed); a named plan "
        "below the bar falls back to the Support winner when its full-batch "
        "delayed is positive, and to identity otherwise.  Selection reads "
        "Support only"
    ),
    "cost": (
        "every Consumer retrain is counted: the identity baselines, the "
        "shortlist evaluations, the mask round's internal per-series retrains "
        "and each delayed reading"
    ),
    "delivery_check": (
        "A5 hits on all three targets and its grounded proposal reason cites "
        "clause ids that are on the card it was served"
    ),
    "direction_check": (
        "per target A5 delayed >= A3 delayed - 0.005, and at least one target "
        "has A5 delayed > A3 delayed + 0.005"
    ),
    "lifecycle_check": (
        "at least one LOCAL_DRAFT forms, with auditable status and evidence "
        "fields"
    ),
    "lifecycle_rule": (
        "the existing rule owns the status.  A target that adopted a "
        "non-identity plan writes an Episode at LOCAL_DRAFT when its Support "
        "gain is at least the material threshold, exactly as e1 does before "
        "delayed evidence arrives, and EPISODE_ONLY otherwise.  Promotion to "
        "LOCAL_ACTIVE needs a delayed probe that did not take part in "
        "selection; this instrument spends its delayed reading on the "
        "adoption gate, so no promotion is attempted and the Draft stands"
    ),
    "overall_verdict": (
        "INTEGRATION_DELIVERS when delivery, direction and lifecycle all hold; "
        "otherwise the failing checks are named, any of RETRIEVAL_MISS, "
        "DIRECTION_LOST, LIFECYCLE_BLOCKED and SCHEMA_BLOCKED, reported "
        "together rather than collapsed into one"
    ),
    "adaptation_rule": (
        "if an interface refuses the card, only the card side may be adapted; "
        "no store, retrieval or lifecycle semantics are changed to make a "
        "check pass"
    ),
    "budget": {
        "llm_calls_total": LLM_CALL_BUDGET_TOTAL,
        "llm_calls_per_arm_target": LLM_CALL_BUDGET_PER_ARM_TARGET,
        "charged_evaluations_per_arm_target": EVALUATION_BUDGET,
        "validation_retries_per_stage": VALIDATION_RETRIES,
    },
    "arm_order": ["%s %s" % pair for pair in ARM_ORDER],
    "circuit_breaker": "stop if the first arm-target produces no payload",
    "no_cross_target_feedback": (
        "the three targets are independent; nothing this run produces is fed "
        "into another target"
    ),
}


# ------------------------------------------------------------- the card side
def _sections(target_id: str, card: Mapping[str, Any]) -> dict[str, str]:
    """The compiled clauses, laid out in the carrier's six sections.

    The clause ids survive verbatim: they are what the delivery check reads,
    and what the Agent is asked to cite.
    """
    target = TARGETS[target_id]
    by_rule: dict[str, list[Mapping[str, Any]]] = {}
    for clause in card["clauses"]:
        by_rule.setdefault(str(clause["clause_id"]).split("-")[0], []).append(
            clause
        )

    def lines(prefix: str) -> str:
        rows = by_rule.get(prefix) or []
        return " ".join(
            "[%s] %s" % (clause["clause_id"], clause["text"]) for clause in rows
        )

    priority = lines("R1")
    risk = lines("R2")
    locality = lines("R3")
    return {
        "WHEN": (
            "A batch of forecast training series processed as one unit under "
            "the %s Consumer structure, where a data-processing program is "
            "chosen for the whole batch before the Consumer is retrained. "
            "Every record behind this card was measured on other cohorts; "
            "everything measured on this one was withheld."
            % target["consumer_variant"]
        ),
        "OBSERVE": (
            "Read the per-series public table already on this Workspace and "
            "the Consumer structure named in the target block. "
            + (locality or "No mask-locality clause was compiled.")
        ),
        "TRY": (
            priority
            or "No priority clause was compiled: the source records did not "
               "put any program above the threshold on this Consumer "
               "structure."
        ),
        "RISK": (
            risk
            or "No risk clause was compiled. The source corpus records a "
               "delayed number only for the plan each cell adopted, so a "
               "program that loses on delayed in two other cohorts almost "
               "never appears; read the absence as missing evidence, not as "
               "safety."
        ),
        "VERIFY": (
            "Believe nothing here until this batch's own Support evaluation "
            "has been spent on it, and the delayed gate has cleared the plan "
            "you name. A clause is guidance and authorizes no execution."
        ),
        "FALLBACK": (
            "If the public observation does not support the clauses, shortlist "
            "on the observation instead, and keep identity: it is always "
            "available and is the incumbent the gate measures against."
        ),
    }


def _card_payload(
    target_id: str, card: Mapping[str, Any], card_text: str,
) -> dict[str, Any]:
    """A ``skill-entry/1`` value.  The store's format, not a new one."""
    sections = _sections(target_id, card)
    target = TARGETS[target_id]
    body = "\n".join(
        "%s: %s" % (name, sections[name].strip())
        for name in ("WHEN", "OBSERVE", "TRY", "RISK", "VERIFY", "FALLBACK")
    )
    return {
        "schema_version": "skill-entry/1",
        "skill_id": SKILL_ID[target_id],
        "skill_kind": "capability",
        "revision": 1,
        "body": body,
        # The same coarse gate the Source-derived General Skill uses.  The
        # observable vocabulary has no cohort and no Consumer-structure
        # feature, so a finer gate cannot be written without inventing one;
        # the finer condition lives in WHEN, as text the Agent reads.
        "observable_applicability": {
            "feature": "task_kind", "op": "==", "value": "forecast",
        },
        "allowed_tools": [],
        "risk_guards": {
            "carrier": "deterministic_recipe_compilation_card",
            "source_class": PROVENANCE_POLICY["source_class"],
            "authorization_scope": PROVENANCE_POLICY["authorization_scope"],
            "advises_the_proposal_stage_only": True,
            "never_supplies_a_candidate": True,
            "requires_target_support": True,
            "grants_confirmation_free_try": False,
            "namespace": NAMESPACE,
            "compiled_for_target": target_id,
            "compiled_cell": "batch:%s|consumer:%s"
            % (target["cohort"], target["consumer_variant"]),
            "leave_one_cohort_out_withheld": str(target["cohort"]),
            "why_this_source_class": list(PROVENANCE_POLICY["why_this_class"]),
            "clause_ids": [
                str(clause["clause_id"]) for clause in card["clauses"]
            ],
            "clauses": [
                {
                    "clause_id": str(clause["clause_id"]),
                    "rule": str(clause["rule"]),
                    "text": str(clause["text"]),
                }
                for clause in card["clauses"]
            ],
            "compiler_artifact": bridge._repo_relative(CARDS_JSON),
            "compiled_card_text": card_text,
            "sections": {
                name: sections[name].strip() for name in sorted(sections)
            },
        },
    }


def _assert_guidance_only(payload: Mapping[str, Any]) -> list[str]:
    """The three carrier guarantees, checked rather than asserted."""
    checks: list[str] = []
    if list(payload["allowed_tools"]):
        raise SystemExit("a guidance card may not carry an allowed tool")
    checks.append("allowed_tools is empty")
    if "Frozen program steps:" in str(payload["body"]):
        raise SystemExit(
            "a guidance card may not carry a frozen program marker"
        )
    checks.append("the body carries no frozen program marker")
    guards = payload["risk_guards"]
    for key in (
        "advises_the_proposal_stage_only", "never_supplies_a_candidate",
        "requires_target_support",
    ):
        if guards.get(key) is not True:
            raise SystemExit("missing carrier guard %r" % key)
    if guards.get("grants_confirmation_free_try") is not False:
        raise SystemExit("a guidance card may not grant a TRY right")
    checks.append(
        "risk_guards declares proposal-stage-only, never-supplies-a-candidate "
        "and requires-target-support, and grants no TRY right"
    )
    return checks


# ---------------------------------------------------------- store registration
def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): canonical_sha256(
            {"bytes": path.read_bytes().hex()}
        )
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    }


def _build_store(slot: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Materialize h0, fork it, add the card, compile, activate.

    Every step is the store's and the compiler's own; nothing here reaches into
    ``methods/ttha/harness/h0`` and nothing is written outside this namespace.
    """
    root = STORE_ROOT / slot / "snapshots"
    store = SnapshotStore(root)
    base = compile_snapshot(H0_ROOT, verify_lock=False)
    parent = store.materialize(base)
    store.set_active(base.runtime_bundle_sha)
    receipt: dict[str, Any] = {
        "slot": slot,
        "store_root": _repo_rel(root),
        "h0_runtime_bundle_sha": base.runtime_bundle_sha,
        "h0_skill_ids": [skill.skill_id for skill in base.skills],
        "card_registered": None,
        "status": None,
    }
    fork = store.fork(parent, "%s-%s" % (NAMESPACE.replace("_", "-"), slot))
    try:
        if payload is not None:
            try:
                entry = load_learned_skill_entry(dict(payload))
            except Exception as exc:  # noqa: BLE001
                receipt.update({
                    "status": "SCHEMA_BLOCKED",
                    "blocked_at_interface": (
                        "SelfEvolvingHarnessTS.contracts.harness."
                        "load_learned_skill_entry"
                    ),
                    "blocked_reason": "%s: %s" % (type(exc).__name__, exc),
                })
                return receipt
            path = fork / "skills" / "learned" / ("%s.json" % entry.skill_id)
            path.write_bytes(canonical_json_bytes(dict(payload)) + b"\n")
            receipt["card_registered"] = {
                "skill_id": entry.skill_id,
                "skill_kind": entry.skill_kind.value,
                "revision": int(entry.revision),
                "allowed_tools": list(entry.allowed_tools),
                "authored_path": path.relative_to(fork).as_posix(),
                "entry_sha256": canonical_sha256(dict(payload)),
                "body_characters": len(str(entry.body)),
            }
        try:
            snapshot = compile_snapshot(fork, verify_lock=False)
        except Exception as exc:  # noqa: BLE001
            receipt.update({
                "status": "SCHEMA_BLOCKED",
                "blocked_at_interface": (
                    "SelfEvolvingHarnessTS.methods.ttha.harness.compiler."
                    "compile_snapshot"
                ),
                "blocked_reason": "%s: %s" % (type(exc).__name__, exc),
            })
            return receipt
        materialized = store.materialize(snapshot, base.runtime_bundle_sha)
        store.set_active(snapshot.runtime_bundle_sha)
    finally:
        store.discard_fork(fork)
    receipt.update({
        "status": "REGISTERED",
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "harness_content_sha": snapshot.harness_content_sha,
        "skill_ids": [skill.skill_id for skill in snapshot.skills],
        "active_pointer": json.loads(
            store.active_path.read_text(encoding="utf-8")
        ),
        "materialized_root": _repo_rel(materialized.root),
        "retrieval_controls": wvc._plain(snapshot.retrieval),
    })
    receipt["_snapshot"] = snapshot
    receipt["_tree"] = _tree(materialized.root)
    return receipt


def _repo_rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def _store_parity(a3: Mapping[str, Any], a5: Mapping[str, Any]) -> dict[str, Any]:
    """Everything outside skills/learned must be byte-identical."""
    left = dict(a3.get("_tree") or {})
    right = dict(a5.get("_tree") or {})
    differing = sorted(
        key for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    )
    # resolved.snapshot.json and snapshot.lock.json restate the skill set, so
    # they move whenever skills/learned does.  Any other file differing would
    # mean the two arms are not running the same Harness.
    expected_prefixes = ("skills/learned/",)
    expected_exact = {"resolved.snapshot.json", "snapshot.lock.json"}
    unexpected = [
        key for key in differing
        if key not in expected_exact
        and not key.startswith(expected_prefixes)
    ]
    return {
        "files_compared": len(set(left) | set(right)),
        "differing_files": differing,
        "unexpected_differences": unexpected,
        "identical_outside_the_skill_library": not unexpected,
        "rule": (
            "the two store trees may differ only in skills/learned and in the "
            "two files whose content sha restates the skill set"
        ),
    }


def register_cards() -> dict[str, Any]:
    """Compile the three cards and register each into its own namespace."""
    if STORE_ROOT.exists():
        shutil.rmtree(STORE_ROOT)
    cards = {
        target_id: bridge.compile_skill_card(target)
        for target_id, target in TARGETS.items()
    }
    texts = {
        target_id: bridge.render_skill_card(card)
        for target_id, card in cards.items()
    }
    payloads = {
        target_id: _card_payload(target_id, cards[target_id], texts[target_id])
        for target_id in TARGETS
    }
    guard_checks = {
        target_id: _assert_guidance_only(payload)
        for target_id, payload in payloads.items()
    }
    slots: dict[str, Any] = {"A3": _build_store("a3", None)}
    for target_id in TARGETS:
        slots[target_id] = _build_store(
            target_id.lower(), payloads[target_id]
        )
    parity = {
        target_id: _store_parity(slots["A3"], slots[target_id])
        for target_id in TARGETS
        if slots[target_id].get("status") == "REGISTERED"
    }
    return {
        "namespace": NAMESPACE,
        "store_root": _repo_rel(STORE_ROOT),
        "h0_source": _repo_rel(H0_ROOT),
        "h0_is_untouched": True,
        "provenance_policy": PROVENANCE_POLICY,
        "carrier_guard_checks": guard_checks,
        "cards": {
            target_id: {
                "status": str(cards[target_id]["status"]),
                "clause_ids": [
                    str(clause["clause_id"])
                    for clause in cards[target_id]["clauses"]
                ],
            }
            for target_id in TARGETS
        },
        "card_payloads": {
            target_id: {
                key: value for key, value in payloads[target_id].items()
                if key != "risk_guards"
            }
            for target_id in TARGETS
        },
        "card_risk_guards": {
            target_id: payloads[target_id]["risk_guards"]
            for target_id in TARGETS
        },
        "slots": {
            key: {
                name: value for name, value in row.items()
                if not name.startswith("_")
            }
            for key, row in slots.items()
        },
        "store_parity": parity,
        "_slots": slots,
        "_cards": cards,
        "_texts": texts,
    }


# --------------------------------------------------------------- the Context
def _public_features(search: Any) -> dict[str, Any]:
    """The Task Context this batch presents, from the real extractor.

    One feature map per training series on its own public prefix, then a
    deterministic aggregate: median for numbers, most common value for strings
    and booleans, ties broken by sort order.  The retrieval gate on these cards
    reads exactly one feature, ``task_kind``, which is the same on every
    series, so the aggregation cannot decide a hit; it is computed and recorded
    so the Context that was resolved against is auditable.
    """
    cutoff = int(search.support[0])
    per_series: dict[str, dict[str, Any]] = {}
    for uid in search.train_uids:
        values = np.asarray(search.values[uid], dtype=np.float64)[:cutoff]
        per_series[uid] = dict(
            extract_public_features(values, task_kind="forecast")
        )
    keys = sorted({key for row in per_series.values() for key in row})
    aggregate: dict[str, Any] = {}
    for key in keys:
        column = [row[key] for row in per_series.values() if key in row]
        if all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in column
        ):
            aggregate[key] = float(statistics.median(column))
        else:
            counted = Counter(column)
            top = max(counted.values())
            aggregate[key] = sorted(
                (item for item, count in counted.items() if count == top),
                key=str,
            )[0]
    return {
        "observation_cutoff": cutoff,
        "series_count": len(per_series),
        "rule": (
            "public features of each training series on its own public prefix "
            "values[uid][:observation_cutoff]; numbers aggregated by median, "
            "strings and booleans by most common value"
        ),
        "features": aggregate,
        "per_series": per_series,
    }


def _retrieve(
    snapshot: Any, features: Mapping[str, Any], expected: str | None,
) -> tuple[Any, dict[str, Any]]:
    """The real three-stage resolution.  The runner adds nothing to it."""
    view = resolve_harness_view(snapshot, dict(features), role="fast")
    served = [skill for skill in view.skills if skill.skill_id == expected]
    record = {
        "resolved_skill_ids": list(view.skill_ids),
        "resolved_memory_ids": list(view.memory_ids),
        "effective_harness_view_sha": view.effective_harness_view_sha,
        "expected_skill_id": expected,
        "hit": bool(served) if expected else None,
        "capability_top_k": wvc._plain(snapshot.retrieval).get("capability"),
        "served_card": None,
    }
    if served:
        skill = served[0]
        guards = dict(skill.risk_guards or {})
        record["served_card"] = {
            "skill_id": skill.skill_id,
            "skill_kind": skill.skill_kind.value,
            "allowed_tools": list(skill.allowed_tools),
            "clause_ids": [str(item) for item in guards.get("clause_ids", ())],
            "body_sha256": canonical_sha256({"body": str(skill.body)}),
            "body": str(skill.body),
            "observable_applicability": wvc._plain(
                skill.observable_applicability
            ),
        }
    return view, record


# ----------------------------------------------------------- the v2 ladder
# The corrected reading of ADOPTION_RULE_V2, ported in the replay slice and
# re-used unchanged here.  Selection is a Support decision; delayed confirms.
LADDER_RULE = bridge.REPLAY_LADDER_RULE
LADDER_V2_CORRESPONDENCE = bridge.REPLAY_V2_CORRESPONDENCE


def _ladder(
    search: Any, *, plans: Sequence[Mapping[str, Any]],
    named: Mapping[str, Any],
) -> dict[str, Any]:
    """Support winner sets the bar; a failed gate walks the full ladder."""
    pool = {
        str(row["program"]): float(row["support_aggregate_gain"])
        for row in plans
        if row.get("full_batch") and str(row["program"]) != IDENTITY
    }
    ranked = sorted(pool, key=lambda op: (-pool[op], bridge._menu_index(op)))
    top = ranked[0] if ranked else None
    winner = top if (top is not None and pool[top] > 0.0) else None

    memo: dict[tuple[str, tuple[str, ...]], Mapping[str, Any]] = {}
    reads: list[dict[str, Any]] = []

    def delayed_of(program: str, excluded: Sequence[str], role: str):
        key = (str(program), tuple(sorted(str(uid) for uid in excluded)))
        fresh = key not in memo
        if fresh:
            memo[key] = search.delayed_gate(str(program), list(key[1]))
        row = memo[key]
        reads.append({
            "program": key[0],
            "excluded_series": list(key[1]),
            "role": role,
            "delayed_aggregate_gain": float(row["aggregate_gain"]),
            "newly_measured": fresh,
        })
        return row

    if winner is None:
        bar = 0.0
        bar_source = "no Support winner, so the bar is identity at zero"
        winner_delayed = None
    else:
        winner_row = delayed_of(winner, [], "bar")
        winner_delayed = float(winner_row["aggregate_gain"])
        bar = max(0.0, winner_delayed)
        bar_source = (
            "max(0, the full-batch delayed of the Support winner `%s`)" % winner
        )
    named_row = delayed_of(
        str(named["program"]), named.get("excluded_series") or (),
        "confirmation",
    )
    named_delayed = float(named_row["aggregate_gain"])
    passed = named_delayed >= bar

    if passed:
        final = {
            "program": str(named["program"]),
            "excluded_series": sorted(
                str(uid) for uid in (named.get("excluded_series") or ())
            ),
        }
        path = "GATE_PASS_ADOPT_NAMED"
        path_text = (
            "the named plan cleared the bar (%+.6f >= %+.6f)"
            % (named_delayed, bar)
        )
        delayed = named_row
    elif winner is not None and float(winner_delayed) > 0.0:
        final = {"program": winner, "excluded_series": []}
        path = "GATE_FAIL_FALLBACK_SUPPORT_WINNER"
        path_text = (
            "the named plan missed the bar by %+.6f, so the ladder fell back "
            "to the Support winner `%s`, whose full-batch delayed is positive"
            % (named_delayed - bar, winner)
        )
        delayed = memo[(winner, ())]
    else:
        final = {"program": IDENTITY, "excluded_series": []}
        path = "GATE_FAIL_FALLBACK_IDENTITY"
        path_text = (
            "the named plan missed the bar by %+.6f and no Support winner has "
            "a positive full-batch delayed, so the ladder fell to identity"
            % (named_delayed - bar)
        )
        delayed = delayed_of(IDENTITY, [], "fallback")
    support = search.support_of_plan(
        final["program"], list(final["excluded_series"])
    )
    return {
        "full_batch_pool": pool,
        "support_ranking": ranked,
        "top_support_program": top,
        "support_winner": winner,
        "support_winner_note": (
            None if winner is not None or top is None else
            "the highest-Support full-batch plan is %r at %+.6f, which is not "
            "positive, so nothing here is a plan a deployer could adopt"
            % (top, pool[top])
        ),
        "support_winner_full_batch_delayed": winner_delayed,
        "bar": bar,
        "bar_source": bar_source,
        "named_plan_delayed_aggregate_gain": named_delayed,
        "named_plan_margin": named_delayed - bar,
        "gate_passed": passed,
        "path": path,
        "path_text": path_text,
        "delayed_reads": reads,
        "delayed_reads_newly_measured": sum(
            1 for row in reads if row["newly_measured"]
        ),
        "final_plan": final,
        "support": support,
        "delayed": delayed,
        "rule": dict(LADDER_RULE),
    }


# ---------------------------------------------------------------- the prompt
SHORTLIST_SCHEMA = bridge.SHORTLIST_SCHEMA
ADOPTION_SCHEMA = bridge.ADOPTION_SCHEMA

SHORTLIST_NOTE = (
    "Choose which programs are worth spending the evaluation budget on. "
    "`shortlist` names at most %d programs from the menu, in the order you "
    "would try them; each one costs one full-batch Support evaluation and the "
    "menu holds %d, so a full scan does not fit. `request_mask_search` asks "
    "for one greedy exclusion round, free of budget, run on whichever "
    "shortlisted program scores highest on Support. If the resolved Harness "
    "you were given carries a retrieved Skill with clause ids, "
    "`skill_clause_use` lists the ones you actually relied on and `reason` "
    "says how; when there is no such Skill, leave it empty and give your "
    "reason from the public observation alone. `reason` is one or two "
    "sentences in public terms. You will see the Support numbers next and "
    "then name the plan; the delayed window is never shown to you."
    % (EVALUATION_BUDGET, len(TREATMENTS))
)

ADOPTION_NOTE = (
    "The measurements are in. Name the plan to adopt: `program` and "
    "`excluded_series` must be exactly one entry of `measured_plans`, which is "
    "everything the instrument measured, `identity` included. `reason` is one "
    "or two sentences in public terms. A named plan whose delayed reading "
    "falls below the bar is not adopted: the adoption ladder falls back to the "
    "highest-Support full-batch plan when its own delayed reading is positive, "
    "and to identity otherwise. You are not shown any delayed reading before "
    "choosing."
)


def _base_input(
    *, target: Mapping[str, Any], window: Mapping[str, Any], search: Any,
    observation: Sequence[Mapping[str, Any]],
    task_spec_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One body for both arms, byte for byte.

    Nothing here names an arm and nothing here carries the card.  Whatever
    guidance reaches the Agent arrives through the resolved Harness that the
    core renders from the retrieved view.
    """
    return {
        "schema_version": "skill-store-integration-input/1",
        "task_spec": _task_spec_observation(target, task_spec_override),
        "target": {
            "target_id": str(target["target_id"]),
            "cohort": str(target["cohort"]),
            "consumer_variant": str(target["consumer_variant"]),
            "cell_key": "batch:%s|consumer:%s"
            % (target["cohort"], target["consumer_variant"]),
            "window_id": str(window["window_id"]),
            "support_origins": list(search.support),
            "delayed_origins": list(search.delayed),
            "observation_cutoff": int(search.support[0]),
            "training_series": list(search.train_uids),
            "evaluation_series_count": len(search.eval_uids),
            "exposure": search.exposure,
        },
        "consumer_structure": {
            "pooled": (
                "one model fitted on the stacked windows of all training "
                "channels"
            ),
            "per_channel": (
                "each training channel fits its own model; every evaluation "
                "channel is predicted by the equal-weight mean of those "
                "channel-wise models"
            ),
        },
        "program_menu": list(TREATMENTS),
        "identity_is_always_available": True,
        "retrieved_guidance": {
            "where_it_comes_from": (
                "any Skill in the resolved Harness of this call was returned "
                "by retrieval for this Task Context; nothing was placed there "
                "by hand"
            ),
            "what_it_authorizes": (
                "guidance for the proposal stage only. A retrieved Skill "
                "cannot supply a candidate and authorizes no execution: the "
                "plan you name is still measured on this batch's own Support "
                "and still has to clear the delayed gate"
            ),
            "may_be_empty": True,
        },
        "evaluation_budget": {
            "charged_evaluations": EVALUATION_BUDGET,
            "menu_size": len(TREATMENTS),
            "mask_round": (
                "free; runs on the highest-Support shortlisted program"
            ),
            "delayed_window": (
                "read after the plan is named and never shown to you"
            ),
            "adoption_ladder": (
                "the Support winner is the highest-Support full-batch plan "
                "measured here, and it sets the bar only if its own Support is "
                "positive; a named plan below the bar falls back to that "
                "Support winner when its delayed reading is positive, and to "
                "identity otherwise"
            ),
        },
        "public_observation": {
            "rule": (
                "public features of each training series on its own public "
                "prefix values[uid][:observation_cutoff]"
            ),
            "rows": [dict(row) for row in observation],
        },
    }


def _make_shortlist_validator(clause_ids: Sequence[str]):
    return bridge._make_shortlist_validator(clause_ids)


# -------------------------------------------------- the TaskSpec observation
# T2 (2026-08-22, task book T2 Part B): the one Observation-surface patch this
# book authorizes.  The public view carried no task or Consumer identity --
# the task was implicitly forecasting (task_kind="forecast" is hardcoded at
# feature extraction and never rendered) and the Consumer was a bare variant
# name with a structural note.  The field below is injected deterministically
# by the runner, never generated by the LLM, and carries no outcome: a
# task_id, a consumer_id, and a one-line statement of what good data means
# for that task.  ``task_spec_override`` exists for the deterministic T2
# prompt materialization only; no live caller passes it.
TASK_QUALITY_SEMANTICS = {
    "forecasting": (
        "good preparation lowers the sMASE of the evaluation-series forecasts"
    ),
    "anomaly_detection": (
        "good preparation keeps the detectable events detectable: it reduces "
        "missed events and false alarms"
    ),
}
CONSUMER_IDS = {
    "pooled": "pooled_ridge_a1",
    "per_channel": "per_channel_ridge_a1",
}


def _task_spec_observation(
    target: Mapping[str, Any],
    override: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """The deterministic TaskSpec observation for the public view."""
    if override is not None:
        return {
            "task_id": str(override["task_id"]),
            "consumer_id": str(override["consumer_id"]),
            "quality_semantics": str(override["quality_semantics"]),
        }
    variant = str(target["consumer_variant"])
    if variant not in CONSUMER_IDS:
        raise ValueError("no consumer_id registered for variant %r" % variant)
    task_id = "forecasting"
    return {
        "task_id": task_id,
        "consumer_id": CONSUMER_IDS[variant],
        "quality_semantics": TASK_QUALITY_SEMANTICS[task_id],
    }


# ------------------------------------------------------------- one arm-target
def _run_arm(
    *, target: Mapping[str, Any], arm: str, window: Mapping[str, Any],
    slot: Mapping[str, Any], expected_skill_id: str | None, llm_budget: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    target_id = str(target["target_id"])
    episode_id = "%s_%s" % (target_id, arm)
    search = bridge.BridgeSearch(
        cohort=str(target["cohort"]),
        consumer_variant=str(target["consumer_variant"]),
        support_origins=window["support_origins"],
        delayed_origins=window["delayed_origins"],
    )
    observation = wvc._observation_table(search)
    context = _public_features(search)
    view, retrieval = _retrieve(
        slot["_snapshot"], context["features"], expected_skill_id,
    )
    served = retrieval.get("served_card") or {}
    clause_ids = [str(item) for item in served.get("clause_ids", ())]
    base = _base_input(
        target=target, window=window, search=search, observation=observation,
    )
    backend = _default_backend_factory(int(llm_budget))
    gateway = wvc.NoToolGateway({"episode_id": episode_id, "arm": arm})
    core = TTHAAgentCore(backend, gateway, model=NF_MODEL, base_url=NF_BASE_URL)

    record: dict[str, Any] = {
        "episode_id": episode_id,
        "target_id": target_id,
        "arm": arm,
        "cohort": str(target["cohort"]),
        "consumer_variant": str(target["consumer_variant"]),
        "window_id": str(window["window_id"]),
        "support_origins": list(search.support),
        "delayed_origins": list(search.delayed),
        "store_slot": slot["slot"],
        "store_runtime_bundle_sha": slot.get("runtime_bundle_sha"),
        "task_context": {
            key: value for key, value in context.items()
            if key != "per_series"
        },
        "task_context_per_series": context["per_series"],
        "retrieval": retrieval,
        "skill_clause_ids_available": clause_ids,
        "public_input_sha256": canonical_sha256(wvc._plain(base)),
        "base_input_field_shas": {
            str(key): canonical_sha256(wvc._plain(value))
            for key, value in base.items()
        },
        "prompt_body": wvc._plain(base),
        "stages": [],
        "shortlist": [],
        "evaluations_used": 0,
        "adopted_plan": None,
        "final_plan": None,
        "support": None,
        "delayed": None,
    }
    shortlist_payload, shortlist_info = wvc._stage(
        core,
        stage="skill_store_shortlist",
        case_id="SSI_%s" % episode_id,
        public_input={**base, "stage_note": SHORTLIST_NOTE},
        harness_view=view,
        schema_name="skill_bridge_shortlist_v1",
        schema=SHORTLIST_SCHEMA,
        validator=_make_shortlist_validator(clause_ids),
    )
    record["stages"].append(shortlist_info)
    record["shortlist_payload"] = wvc._plain(shortlist_payload)
    if shortlist_payload is None:
        record["llm_calls"] = int(backend.calls)
        record["instrument"] = search.accounting()
        record["consumer_retrains_total"] = int(search.retrains)
        record["wall_seconds"] = time.perf_counter() - started
        return record

    shortlist = [str(item) for item in shortlist_payload["shortlist"]]
    wants_mask = bool(shortlist_payload["request_mask_search"])
    cited = [str(item) for item in shortlist_payload.get("skill_clause_use", ())]
    support_results = {
        program: search.full_batch_support(program) for program in shortlist
    }
    mask_result = None
    if wants_mask:
        best = max(
            shortlist,
            key=lambda program: (
                support_results[program]["aggregate_gain"],
                -shortlist.index(program),
            ),
        )
        mask_result = search.mask_search(best)
    plans, mask_note = bridge._measured_plans(
        shortlist=shortlist, support_results=support_results,
        mask_result=mask_result,
    )
    record.update({
        "shortlist": shortlist,
        "request_mask_search": wants_mask,
        "skill_clause_use": cited,
        "shortlist_reason": str(shortlist_payload.get("reason", "")),
        "support_results": support_results,
        "mask_search": wvc._plain(mask_result),
        "measured_plans": plans,
        "measured_plans_note": mask_note,
        "evaluations_used": int(search.support_evaluations_charged),
    })
    print(
        "SSI %s hit=%s shortlist=%s mask=%s cited=%s"
        % (episode_id, retrieval.get("hit"), shortlist, wants_mask, cited),
        flush=True,
    )

    adoption_payload, adoption_info = wvc._stage(
        core,
        stage="skill_store_adoption",
        case_id="SSI_%s" % episode_id,
        public_input={
            **base,
            "stage_note": ADOPTION_NOTE,
            "your_shortlist": list(shortlist),
            "measured_plans": [dict(row) for row in plans],
            "measured_plans_note": mask_note,
            "evaluations_spent": int(record["evaluations_used"]),
        },
        harness_view=view,
        schema_name="budgeted_adoption_v1",
        schema=ADOPTION_SCHEMA,
        validator=wvc._make_adoption_validator(
            shortlist=shortlist, mask_result=record.get("mask_search"),
        ),
    )
    record["stages"].append(adoption_info)
    record["adoption_payload"] = wvc._plain(adoption_payload)
    record["llm_calls"] = int(backend.calls)
    if adoption_payload is None:
        record["instrument"] = search.accounting()
        record["consumer_retrains_total"] = int(search.retrains)
        record["wall_seconds"] = time.perf_counter() - started
        return record

    named = {
        "program": str(adoption_payload["program"]),
        "excluded_series": sorted(
            str(uid) for uid in adoption_payload.get("excluded_series", ())
        ),
    }
    ladder = _ladder(search, plans=plans, named=named)
    final_plan = ladder["final_plan"]
    support_gain = float(ladder["support"]["aggregate_gain"])
    delayed_gain = float(ladder["delayed"]["aggregate_gain"])
    reference = float(window["reference_delayed_aggregate_gain"])
    if final_plan["program"] == IDENTITY:
        relation = RELATION_ABSTAIN
    elif support_gain > 0.0 and delayed_gain > 0.0:
        relation = RELATION_POSITIVE
    elif (support_gain > 0.0) != (delayed_gain > 0.0):
        relation = RELATION_CONFLICT
    else:
        relation = RELATION_NEGATIVE
    record.update({
        "adopted_plan": named,
        "adoption_reason": str(adoption_payload.get("reason", "")),
        "adoption_ladder": {
            key: value for key, value in ladder.items()
            if key not in ("support", "delayed")
        },
        "final_plan": final_plan,
        "support": ladder["support"],
        "delayed": ladder["delayed"],
        "relation": relation,
        "reference_delayed_aggregate_gain": reference,
        "reference_plan": dict(window["reference_plan"]),
        "capture_ratio": (delayed_gain / reference if reference else None),
        "matches_reference_plan": bool(
            final_plan["program"] == str(window["reference_plan"]["program"])
            and final_plan["excluded_series"]
            == sorted(
                str(uid) for uid in window["reference_plan"]["excluded_series"]
            )
        ),
        "instrument": search.accounting(),
        "consumer_retrains_total": int(search.retrains),
        "wall_seconds": time.perf_counter() - started,
    })
    print(
        "SSI %s final %s minus %s | support %+.6f delayed %+.6f | bar %+.6f "
        "%s | evals %d retrains %d llm %d"
        % (
            episode_id, final_plan["program"],
            ", ".join(final_plan["excluded_series"]) or "nothing",
            support_gain, delayed_gain, float(ladder["bar"]), ladder["path"],
            record["evaluations_used"], search.retrains, record["llm_calls"],
        ),
        flush=True,
    )
    return record


# ----------------------------------------------------------------- lifecycle
AUDIT = {
    "provenance": EXPERIENCE_PROVENANCE,
    "counts_as_unguided_exploration": False,
    "audit_note": (
        "engineering measurement of the Skill-store channel; not authorization "
        "evidence and not an unguided probe. The registered card is guidance "
        "in a namespace of its own, carries no program and no tool, and grants "
        "no TRY right"
    ),
}


def _lifecycle(record: Mapping[str, Any]) -> dict[str, Any]:
    """Write the adopted plan through the existing Experience lifecycle.

    The status rule is e1's, unchanged: an Episode whose Support gain reaches
    the material threshold is a Draft, everything else is Episode-only.  What
    this instrument cannot do is promote: e1 promotes a Draft to LOCAL_ACTIVE
    from a delayed probe that did not take part in selection, and here the
    delayed reading is what the adoption gate spent.  So the Draft stands and
    no promotion is attempted.
    """
    plan = record.get("final_plan")
    if plan is None or record.get("support") is None:
        return {
            "status": None,
            "written": False,
            "reason": "this arm-target produced no adopted plan",
        }
    support_gain = float(record["support"]["aggregate_gain"])
    delayed_gain = float(record["delayed"]["aggregate_gain"])
    non_identity = str(plan["program"]) != IDENTITY
    support_material = support_gain >= MATERIAL_THRESHOLD
    delayed_positive = delayed_gain > 0.0
    draft = bool(non_identity and support_material and delayed_positive)
    status = STATUS_LOCAL_DRAFT if draft else STATUS_EPISODE_ONLY
    episode = build_episode(
        episode_id="ssi_%s" % str(record["episode_id"]).lower(),
        task_consumer_key="batch:%s|consumer:%s"
        % (record["cohort"], record["consumer_variant"]),
        domain_namespace=str(record["cohort"]),
        context_summary={
            "cohort": {"cohort_name": str(record["cohort"])},
            "local_pattern": {
                "consumer_variant": str(record["consumer_variant"]),
                "window_id": str(record["window_id"]),
                "arm": str(record["arm"]),
                "retrieved_skill_ids": list(
                    (record.get("retrieval") or {}).get(
                        "resolved_skill_ids", ()
                    )
                ),
            },
            "program_geometry": {
                "program": str(plan["program"]),
                "excluded_count": len(plan["excluded_series"]),
                "evaluations_used": int(record["evaluations_used"]),
                "consumer_retrains": int(record["consumer_retrains_total"]),
            },
        },
        workflow_signature=workflow_signature_of(
            () if not non_identity else ({"op": str(plan["program"])},)
        ),
        support_response={
            "gain": support_gain,
            "window": "support",
            "program": str(plan["program"]),
            "excluded_series": list(plan["excluded_series"]),
            "accepted": support_material,
            "block_origins": list(record["support_origins"]),
            "harmed_eval_series_count": int(
                record["support"]["harmed_eval_series_count"]
            ),
            "skill_clause_use": list(record.get("skill_clause_use") or []),
            **AUDIT,
        },
        delayed_response={
            "evaluated": True,
            "gain": delayed_gain,
            "se_block": None,
            "gain_over_se": None,
            "block_origins": list(record["delayed_origins"]),
            "took_part_in_selection": True,
            "why_not_delayed_level_evidence": (
                "this reading set the adoption bar and decided the plan, so it "
                "is in-selection; the promotion rule wants a delayed probe "
                "that did not"
            ),
            "harmed_eval_series_count": int(
                record["delayed"]["harmed_eval_series_count"]
            ),
            **AUDIT,
        },
        relation=str(record["relation"]),
        evidence_level=EVIDENCE_SUPPORT,
        local_status=status,
        evidence_refs=(EXPERIENCE_PROVENANCE, PROTOCOL_VERSION),
    )
    return {
        "status": status,
        "written": True,
        "is_draft": draft,
        "conditions": {
            "adopted_non_identity": non_identity,
            "support_at_or_above_material_threshold": support_material,
            "support_aggregate_gain": support_gain,
            "material_threshold": MATERIAL_THRESHOLD,
            "delayed_positive": delayed_positive,
            "delayed_aggregate_gain": delayed_gain,
        },
        "evidence_level": EVIDENCE_SUPPORT,
        "relation": str(record["relation"]),
        "promotion_attempted": False,
        "promotion_blocked_by": (
            "LOCAL_ACTIVE needs a delayed probe that did not take part in "
            "selection; this instrument spends its delayed reading on the "
            "adoption gate, so the Draft stands"
        ),
        "rule_source": (
            "evaluation/functional/task_episode_harness/e1.py, the Episode "
            "built before delayed evidence arrives and _update_delayed that "
            "promotes it"
        ),
        "episode": episode.to_dict(),
    }


# ------------------------------------------------------------------ verdicts
def _target_verdict(
    *, target_id: str, a3: Mapping[str, Any] | None,
    a5: Mapping[str, Any] | None, registration: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> dict[str, Any]:
    slot = registration["slots"].get(target_id) or {}
    row: dict[str, Any] = {
        "target_id": target_id,
        "registration_status": slot.get("status"),
        "skill_id": SKILL_ID[target_id],
    }
    if slot.get("status") != "REGISTERED":
        row.update({
            "delivery": "SCHEMA_BLOCKED",
            "blocked_at_interface": slot.get("blocked_at_interface"),
            "blocked_reason": slot.get("blocked_reason"),
            "direction": None,
            "paired_delayed_delta": None,
            "lifecycle_status": None,
        })
        return row
    hit = bool((a5 or {}).get("retrieval", {}).get("hit"))
    available = [str(item) for item in (a5 or {}).get(
        "skill_clause_ids_available", ()
    )]
    cited = [str(item) for item in (a5 or {}).get("skill_clause_use", ())]
    grounded = sorted(set(cited) & set(available))
    if not hit:
        delivery = "RETRIEVAL_MISS"
    elif not grounded:
        delivery = "CLAUSE_NOT_CITED"
    else:
        delivery = "DELIVERED"
    readable = (
        a3 is not None and a5 is not None
        and a3.get("delayed") is not None and a5.get("delayed") is not None
    )
    if not readable:
        row.update({
            "delivery": delivery,
            "clauses_available": available,
            "clauses_cited_by_a5": cited,
            "clauses_grounded": grounded,
            "direction": "UNREADABLE",
            "paired_delayed_delta": None,
            "lifecycle_status": lifecycle.get("status"),
        })
        return row
    a3_delayed = float(a3["delayed"]["aggregate_gain"])
    a5_delayed = float(a5["delayed"]["aggregate_gain"])
    delta = a5_delayed - a3_delayed
    if delta > QUALITY_DELTA_THRESHOLD:
        direction = "A5_WINS"
    elif delta < -QUALITY_DELTA_THRESHOLD:
        direction = "A5_LOSES"
    else:
        direction = "TIE"
    row.update({
        "delivery": delivery,
        "retrieval_hit": hit,
        "resolved_skill_ids": list(
            a5["retrieval"].get("resolved_skill_ids", ())
        ),
        "clauses_available": available,
        "clauses_cited_by_a5": cited,
        "clauses_grounded": grounded,
        "a5_shortlist_reason": str(a5.get("shortlist_reason", "")),
        "a5_adoption_reason": str(a5.get("adoption_reason", "")),
        "direction": direction,
        "non_inferior": a5_delayed >= a3_delayed - QUALITY_DELTA_THRESHOLD,
        "paired_delayed_delta": delta,
        "a3_final_plan": a3["final_plan"],
        "a5_final_plan": a5["final_plan"],
        "a3_named_plan": a3["adopted_plan"],
        "a5_named_plan": a5["adopted_plan"],
        "a3_support_aggregate_gain": float(a3["support"]["aggregate_gain"]),
        "a5_support_aggregate_gain": float(a5["support"]["aggregate_gain"]),
        "a3_delayed_aggregate_gain": a3_delayed,
        "a5_delayed_aggregate_gain": a5_delayed,
        "a3_capture_ratio": a3.get("capture_ratio"),
        "a5_capture_ratio": a5.get("capture_ratio"),
        "a3_ladder_path": (a3.get("adoption_ladder") or {}).get("path"),
        "a5_ladder_path": (a5.get("adoption_ladder") or {}).get("path"),
        "a3_consumer_retrains": int(a3["consumer_retrains_total"]),
        "a5_consumer_retrains": int(a5["consumer_retrains_total"]),
        "a3_evaluations_used": int(a3["evaluations_used"]),
        "a5_evaluations_used": int(a5["evaluations_used"]),
        "lifecycle_status": lifecycle.get("status"),
        "lifecycle_is_draft": lifecycle.get("is_draft"),
        "reason": (
            "delivery %s; paired delayed delta %+.6f (A5 %+.6f - A3 %+.6f) "
            "against %.3f; lifecycle %s"
            % (
                delivery, delta, a5_delayed, a3_delayed,
                QUALITY_DELTA_THRESHOLD, lifecycle.get("status"),
            )
        ),
    })
    return row


def _overall(per_target: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    blocked = sorted(
        key for key, row in per_target.items()
        if row.get("delivery") == "SCHEMA_BLOCKED"
    )
    missed = sorted(
        key for key, row in per_target.items()
        if row.get("delivery") == "RETRIEVAL_MISS"
    )
    uncited = sorted(
        key for key, row in per_target.items()
        if row.get("delivery") == "CLAUSE_NOT_CITED"
    )
    deliveries = [row.get("delivery") for row in per_target.values()]
    delivery_ok = all(item == "DELIVERED" for item in deliveries)
    readable = [
        row for row in per_target.values()
        if row.get("paired_delayed_delta") is not None
    ]
    non_inferior = [row for row in readable if row.get("non_inferior")]
    winners = sorted(
        key for key, row in per_target.items() if row.get("direction") == "A5_WINS"
    )
    direction_ok = bool(
        len(readable) == len(per_target)
        and len(non_inferior) == len(per_target)
        and winners
    )
    drafts = sorted(
        key for key, row in per_target.items()
        if row.get("lifecycle_status") == STATUS_LOCAL_DRAFT
    )
    lifecycle_ok = bool(drafts)
    blocked_layers: list[dict[str, Any]] = []
    if blocked:
        blocked_layers.append({
            "layer": "skill store / card schema",
            "interface": per_target[blocked[0]].get("blocked_at_interface"),
            "targets": blocked,
        })
    if missed:
        blocked_layers.append({
            "layer": "retrieval",
            "interface": (
                "SelfEvolvingHarnessTS.methods.ttha.retrieval."
                "resolve_harness_view"
            ),
            "targets": missed,
        })
    if uncited:
        blocked_layers.append({
            "layer": "Fast reading of the resolved Harness",
            "interface": (
                "SelfEvolvingHarnessTS.methods.ttha.agent_core."
                "TTHAAgentCore._messages"
            ),
            "targets": uncited,
        })
    if not lifecycle_ok:
        blocked_layers.append({
            "layer": "Target-local lifecycle",
            "interface": (
                "SelfEvolvingHarnessTS.methods.ttha.experience_memory."
                "build_episode"
            ),
            "targets": sorted(per_target),
        })
    labels: list[str] = []
    if blocked:
        labels.append("SCHEMA_BLOCKED")
    if missed:
        labels.append("RETRIEVAL_MISS")
    if uncited:
        labels.append("SKILL_NOT_CITED")
    if not direction_ok:
        labels.append("DIRECTION_LOST")
    if not lifecycle_ok:
        labels.append("LIFECYCLE_BLOCKED")
    if delivery_ok and direction_ok and lifecycle_ok:
        verdict = "INTEGRATION_DELIVERS"
        reason = (
            "the card was retrieved and cited on every target, every target is "
            "non-inferior on delayed with A5 ahead on %s, and %s formed a "
            "LOCAL_DRAFT"
            % (", ".join(winners), ", ".join(drafts))
        )
    else:
        verdict = " + ".join(labels) if labels else "INTEGRATION_DELIVERS"
        reason = "; ".join(
            part for part in (
                ("registration blocked on %s" % ", ".join(blocked))
                if blocked else "",
                ("retrieval missed on %s" % ", ".join(missed))
                if missed else "",
                ("no card clause was cited on %s" % ", ".join(uncited))
                if uncited else "",
                (
                    "direction check failed: %d of %d targets non-inferior, "
                    "winners %s"
                    % (
                        len(non_inferior), len(per_target),
                        ", ".join(winners) or "none",
                    )
                ) if not direction_ok else "",
                "no LOCAL_DRAFT formed" if not lifecycle_ok else "",
            ) if part
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "checks": {
            "delivery": {
                "passed": delivery_ok,
                "per_target": {
                    key: row.get("delivery") for key, row in per_target.items()
                },
            },
            "direction": {
                "passed": direction_ok,
                "non_inferior_count": len(non_inferior),
                "targets": len(per_target),
                "winners": winners,
                "per_target": {
                    key: row.get("direction") for key, row in per_target.items()
                },
            },
            "lifecycle": {
                "passed": lifecycle_ok,
                "drafts": drafts,
                "per_target": {
                    key: row.get("lifecycle_status")
                    for key, row in per_target.items()
                },
            },
        },
        "blocked_layers": blocked_layers,
    }


def _prompt_parity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, dict[str, Any]] = {}
    for row in records:
        by_target.setdefault(str(row["target_id"]), {})[str(row["arm"])] = row
    per_target: dict[str, Any] = {}
    for target_id, arms in by_target.items():
        a3, a5 = arms.get("A3"), arms.get("A5")
        if a3 is None or a5 is None:
            per_target[target_id] = {"comparable": False}
            continue
        same = str(a3["public_input_sha256"]) == str(a5["public_input_sha256"])
        differing = sorted(
            key for key in set(a3["base_input_field_shas"])
            | set(a5["base_input_field_shas"])
            if a3["base_input_field_shas"].get(key)
            != a5["base_input_field_shas"].get(key)
        )
        per_target[target_id] = {
            "comparable": True,
            "public_input_sha256": str(a3["public_input_sha256"]),
            "identical": bool(same and not differing),
            "differing_fields": differing,
        }
    return {
        "scope": (
            "the whole public input body, hashed once and field by field. The "
            "arms differ only in the store they resolve against"
        ),
        "all_targets_identical": all(
            row.get("identical") for row in per_target.values()
        ),
        "per_target": per_target,
    }


# ---------------------------------------------------------------------- run
def run(*, dry_run: bool = False) -> int:
    started = time.perf_counter()
    registration = register_cards()
    for target_id, slot in registration["_slots"].items():
        print(
            "SSI register %-3s %-16s %s"
            % (
                target_id, slot.get("status"),
                (slot.get("card_registered") or {}).get("skill_id") or "-",
            ),
            flush=True,
        )
    windows = {
        target_id: bridge._target_window(target)
        for target_id, target in TARGETS.items()
    }
    records: list[dict[str, Any]] = []
    lifecycles: dict[str, Any] = {}
    llm_used = 0
    stopped: str | None = None
    for target_id, arm in ARM_ORDER:
        slot = registration["_slots"]["A3" if arm == "A3" else target_id]
        if slot.get("status") != "REGISTERED":
            print(
                "SSI skip %s_%s: store slot is %s"
                % (target_id, arm, slot.get("status")), flush=True,
            )
            continue
        remaining = LLM_CALL_BUDGET_TOTAL - llm_used
        if remaining <= 0:
            stopped = "the total LLM budget of %d was spent" % (
                LLM_CALL_BUDGET_TOTAL
            )
            break
        record = _run_arm(
            target=TARGETS[target_id],
            arm=arm,
            window=windows[target_id],
            slot=slot,
            expected_skill_id=(SKILL_ID[target_id] if arm == "A5" else None),
            llm_budget=min(LLM_CALL_BUDGET_PER_ARM_TARGET, remaining),
        )
        llm_used += int(record.get("llm_calls") or 0)
        life = _lifecycle(record)
        lifecycles[str(record["episode_id"])] = life
        record["lifecycle"] = {
            key: value for key, value in life.items() if key != "episode"
        }
        records.append(record)
        if len(records) == 1 and record.get("final_plan") is None:
            stopped = (
                "the first arm-target produced no adopted plan; the run "
                "stopped before spending more budget"
            )
            break

    by_id = {str(row["episode_id"]): row for row in records}
    per_target: dict[str, Any] = {}
    for target_id in TARGETS:
        a5_life = lifecycles.get("%s_A5" % target_id) or {}
        per_target[target_id] = _target_verdict(
            target_id=target_id,
            a3=by_id.get("%s_A3" % target_id),
            a5=by_id.get("%s_A5" % target_id),
            registration=registration,
            lifecycle=a5_life,
        )
        per_target[target_id]["a3_lifecycle_status"] = (
            lifecycles.get("%s_A3" % target_id) or {}
        ).get("status")
    overall = _overall(per_target)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "does the compiled recipe Skill survive the real channel: store "
            "registration, retrieval by Task Context, Fast reading the "
            "resolved Harness, and the adopted plan entering the Target-local "
            "lifecycle"
        ),
        "not_authorization_evidence": (
            "the registered card carries no program and no tool, is scoped to "
            "a store namespace of its own, and grants no TRY right; adoption "
            "still passes this batch's own Support evaluation and the delayed "
            "gate. No Skill is promoted and no Slow path runs"
        ),
        "not_new_transfer_evidence": (
            "the same three targets the bridge run already measured; this "
            "slice measures channel loss, not transfer"
        ),
        "overall_verdict": overall["verdict"],
        "overall_verdict_reason": overall["reason"],
        "checks": overall["checks"],
        "blocked_layers": overall["blocked_layers"],
        "per_target": per_target,
        "pre_registered": PRE_REGISTERED,
        "provenance_policy": PROVENANCE_POLICY,
        "registration": {
            key: value for key, value in registration.items()
            if not key.startswith("_")
        },
        "adoption_ladder_rule": dict(LADDER_RULE),
        "adoption_ladder_v2_correspondence": LADDER_V2_CORRESPONDENCE,
        "prompt_parity_check": _prompt_parity(records),
        "cost_report": bridge._cost_report(records),
        "lifecycle_records": {
            key: value for key, value in lifecycles.items()
        },
        "experience_entries_written": [
            value["episode"] for value in lifecycles.values()
            if value.get("episode")
        ],
        "experience_provenance": EXPERIENCE_PROVENANCE,
        "skill_card_rendered_text": registration["_texts"],
        "model": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "target_windows": {
            target_id: {
                key: value for key, value in window.items()
                if key != "reference"
            }
            for target_id, window in windows.items()
        },
        "llm_call_count": llm_used,
        "llm_call_budget_total": LLM_CALL_BUDGET_TOTAL,
        "stopped_early": stopped,
        "arm_targets": records,
        "wall_seconds": time.perf_counter() - started,
    }
    if dry_run:
        print(json.dumps(
            {
                "overall": payload["overall_verdict"],
                "per_target": {
                    key: {
                        "delivery": row.get("delivery"),
                        "direction": row.get("direction"),
                        "delta": row.get("paired_delayed_delta"),
                        "lifecycle": row.get("lifecycle_status"),
                    }
                    for key, row in per_target.items()
                },
            },
            indent=2, ensure_ascii=False, default=str,
        ))
        return 0
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("overall", payload["overall_verdict"], flush=True)
    print("llm_calls", llm_used, flush=True)
    return 0


# ------------------------------------------------------------------- report
def _plan_label(plan: Mapping[str, Any] | None) -> str:
    if not plan:
        return "--"
    excluded = [str(uid) for uid in (plan.get("excluded_series") or [])]
    if not excluded:
        return "`%s` full batch" % plan["program"]
    return "`%s` minus %s" % (plan["program"], ", ".join(sorted(excluded)))


def _markdown(payload: Mapping[str, Any]) -> str:
    per_target = payload["per_target"]
    registration = payload["registration"]
    policy = payload["provenance_policy"]
    parity = payload["prompt_parity_check"]
    cost = payload["cost_report"]
    lines = [
        "# recipe Skill through the real store, retrieval and lifecycle",
        "",
        "**Overall: `%s`** -- %s."
        % (payload["overall_verdict"], payload["overall_verdict_reason"]),
        "",
        "The bridge run put the compiled card into the prompt by hand.  This "
        "slice keeps the signal and the three targets fixed and changes only "
        "the channel: each leave-one-cohort-out card is registered into the "
        "Skill store under the store's own `skill-entry/1` schema, Fast reads "
        "whatever `resolve_harness_view` returns for the Task Context, and the "
        "adopted plan is written through the existing Experience lifecycle.  "
        "The two arms send byte-identical public input; the only difference is "
        "the store they resolve against.",
        "",
        "**It is not new evidence that the signal transfers.**  These are the "
        "same three targets the bridge already measured, so what is measured "
        "here is channel loss.",
        "",
        "**Engineering integration measurement, not authorization evidence.**",
        "",
        "## Source-class and authorization, as registered",
        "",
        "Every card is registered with `source_class = \"%s\"` and "
        "`authorization_scope = \"%s\"`.  The three reasons that class is "
        "claimed, recorded on each entry's `risk_guards`:"
        % (policy["source_class"], policy["authorization_scope"]),
        "",
    ]
    for reason in policy["why_this_class"]:
        lines.append("- %s." % reason)
    lines += [
        "",
        "What GUIDANCE means here: %s." % policy["what_guidance_means"],
        "",
        "What is **not** granted: %s." % policy["what_is_not_granted"],
        "",
        "The carrier guarantees are checked at registration, not asserted: "
        + "; ".join(policy["carrier_guards_asserted_at_registration"]) + ".",
        "",
        "Leave-one-cohort-out: %s." % policy["loco"],
        "",
        "## Registration",
        "",
        "| slot | status | skill id | runtime bundle | skills in snapshot |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for key, slot in registration["slots"].items():
        lines.append(
            "| `%s` | `%s` | `%s` | `%s` | %s |"
            % (
                key, slot.get("status"),
                (slot.get("card_registered") or {}).get("skill_id") or "--",
                str(slot.get("runtime_bundle_sha") or "--")[:12],
                len(slot.get("skill_ids") or []),
            )
        )
    lines += [
        "",
        "Store namespace `%s` under `%s`; `%s` is read and never written."
        % (
            registration["namespace"], registration["store_root"],
            registration["h0_source"],
        ),
        "",
    ]
    for key, row in (registration.get("store_parity") or {}).items():
        lines.append(
            "- `%s` vs the empty-store arm: %d files compared, differing %s -- "
            "%s"
            % (
                key, row["files_compared"],
                ", ".join("`%s`" % item for item in row["differing_files"]),
                "nothing unexpected"
                if row["identical_outside_the_skill_library"]
                else "**unexpected: %s**" % row["unexpected_differences"],
            )
        )
    lines += [
        "",
        "## Per target",
        "",
        "| target | delivery | retrieved | clauses cited | A3 plan | A3 "
        "delayed | A5 plan | A5 delayed | delta | direction | lifecycle |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for key in sorted(per_target):
        row = per_target[key]
        if row.get("paired_delayed_delta") is None:
            lines.append(
                "| `%s` | `%s` | %s | %s | -- | -- | -- | -- | -- | `%s` | `%s` |"
                % (
                    key, row.get("delivery"), row.get("retrieval_hit"),
                    ", ".join(row.get("clauses_grounded") or []) or "--",
                    row.get("direction"), row.get("lifecycle_status"),
                )
            )
            continue
        lines.append(
            "| `%s` | `%s` | `%s` | %s | %s | %+.6f | %s | %+.6f | **%+.6f** | "
            "`%s` | `%s` |"
            % (
                key, row["delivery"],
                (row.get("resolved_skill_ids") or ["--"])[-1],
                ", ".join(row.get("clauses_grounded") or []) or "--",
                _plan_label(row["a3_final_plan"]),
                row["a3_delayed_aggregate_gain"],
                _plan_label(row["a5_final_plan"]),
                row["a5_delayed_aggregate_gain"],
                row["paired_delayed_delta"], row["direction"],
                row["lifecycle_status"],
            )
        )
    lines += [
        "",
        "Capture against each target's full-search reference: "
        + "; ".join(
            "%s A3 %s -> A5 %s"
            % (
                key,
                "--" if per_target[key].get("a3_capture_ratio") is None
                else "%.3f" % per_target[key]["a3_capture_ratio"],
                "--" if per_target[key].get("a5_capture_ratio") is None
                else "%.3f" % per_target[key]["a5_capture_ratio"],
            )
            for key in sorted(per_target)
        )
        + ".",
        "",
        "### What A5 cited",
        "",
    ]
    for key in sorted(per_target):
        row = per_target[key]
        if not row.get("a5_shortlist_reason"):
            continue
        lines.append(
            "- **%s** cited %s: \"%s\""
            % (
                key, ", ".join(row.get("clauses_grounded") or []) or "nothing",
                str(row["a5_shortlist_reason"]).replace("\n", " "),
            )
        )
    lines += [
        "",
        "## Checks",
        "",
        "| check | passed | detail |",
        "| --- | --- | --- |",
    ]
    checks = payload["checks"]
    lines.append(
        "| delivery | %s | %s |"
        % (
            checks["delivery"]["passed"],
            json.dumps(checks["delivery"]["per_target"], sort_keys=True),
        )
    )
    lines.append(
        "| direction | %s | %d of %d non-inferior, winners %s |"
        % (
            checks["direction"]["passed"],
            checks["direction"]["non_inferior_count"],
            checks["direction"]["targets"],
            ", ".join(checks["direction"]["winners"]) or "none",
        )
    )
    lines.append(
        "| lifecycle | %s | drafts %s |"
        % (
            checks["lifecycle"]["passed"],
            ", ".join(checks["lifecycle"]["drafts"]) or "none",
        )
    )
    if payload.get("blocked_layers"):
        lines += ["", "### Where it stuck", ""]
        for item in payload["blocked_layers"]:
            lines.append(
                "- **%s** -- `%s` on %s"
                % (
                    item["layer"], item.get("interface"),
                    ", ".join(item.get("targets") or []) or "every target",
                )
            )
    lines += [
        "",
        "## Lifecycle",
        "",
        "| arm-target | status | relation | evidence level | promotion |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in sorted(payload["lifecycle_records"]):
        row = payload["lifecycle_records"][key]
        lines.append(
            "| `%s` | `%s` | `%s` | `%s` | %s |"
            % (
                key, row.get("status"), row.get("relation"),
                row.get("evidence_level"),
                "not attempted" if row.get("promotion_attempted") is False
                else "--",
            )
        )
    first_draft = next(
        (
            row for row in payload["lifecycle_records"].values()
            if row.get("is_draft")
        ),
        None,
    )
    if first_draft is not None:
        lines += [
            "",
            "Promotion is not attempted: %s."
            % first_draft["promotion_blocked_by"],
        ]
    lines += [
        "",
        "## Cost and parity",
        "",
        "Consumer retrains: A3 %d, A5 %d, %d in all.  %s"
        % (
            cost["arm_totals"]["A3"], cost["arm_totals"]["A5"],
            cost["arm_totals"]["A3"] + cost["arm_totals"]["A5"],
            "First delayed-positive adoption: "
            + "; ".join(
                "%s at `%s` after %s retrains"
                % (
                    arm, (row or {}).get("at_episode"),
                    (row or {}).get("cumulative_consumer_retrains"),
                )
                for arm, row in cost[
                    "cumulative_retrains_to_first_delayed_positive_adoption"
                ].items()
            )
            + ".",
        ),
        "",
        "Prompt parity: %s.  %s"
        % (
            "identical on every target"
            if parity["all_targets_identical"]
            else "**not identical**",
            json.dumps(
                {
                    key: row.get("differing_fields")
                    for key, row in parity["per_target"].items()
                },
                sort_keys=True,
            ),
        ),
        "",
        "LLM calls %d of %d.  %s"
        % (
            payload["llm_call_count"], payload["llm_call_budget_total"],
            payload.get("stopped_early") or "No early stop.",
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--register-only", action="store_true",
        help="compile and register the three cards, then stop (0 LLM)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run everything but print the verdict instead of writing",
    )
    args = parser.parse_args(argv)
    if args.register_only:
        registration = register_cards()
        for key, slot in registration["slots"].items():
            print("=== %s: %s" % (key, slot.get("status")))
            print(json.dumps(slot, indent=2, ensure_ascii=False, default=str))
        for key, row in (registration.get("store_parity") or {}).items():
            print("parity %s: %s" % (key, json.dumps(row, default=str)))
        return 0
    return run(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
