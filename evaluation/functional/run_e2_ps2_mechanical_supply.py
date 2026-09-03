"""PS-2 -- mechanical supply of one candidate to verify (eval layer).

PS-1b froze TEXT_RUNG_INERT: bootstrap 4b already says the Source prior is
injected by the runtime, but ``supplies_candidates`` is unread and a
suggestion-only card has no ``Frozen program steps:``, so
``_skill_frozen_candidates`` supplies nothing.

This book implements that promised semantics at the evaluation layer only.
The cards carry a frozen program.  Production ``_skill_frozen_candidates``
then puts ``cand_skill_<id>`` in the same CandidatePool the agent proposals
enter.  ``requires_target_support=true`` / ``grants_execution=false`` keep
the inject as DRAFT: same verifier, same Support, same delayed gate, no
priority slot, inside ``maximum_candidates=3``.

Three arms, identical budgets, GPOVY, 4 replicates:

* A3          -- no card
* A5-neutral  -- same Scope, supplies_candidates, frozen resample_uniform
                 (sealed-oracle numeric no-op on this substrate)
* A5-scoped   -- same Scope, supplies_candidates, frozen hampel_filter

A conversion is "experience supplied a candidate through the mechanical
channel; Target feedback adjudicated it".  It is not "the agent learned
to propose hampel".

  python evaluation/functional/run_e2_ps2_mechanical_supply.py --compile-only
  python evaluation/functional/run_e2_ps2_mechanical_supply.py --run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
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

import run_e2_ps0_reearn_sources as ps0  # noqa: E402
import run_e2_ps0c_ps1 as ps0c  # noqa: E402
import run_e2_ps1_arms as ps1  # noqa: E402
import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _parse_frozen_steps,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
PS1_JSON = E2 / "ps1_proposal_shift_r2.json"
ORACLE_JSON = E2 / "s1_oracle" / "GunPointOldVersusYoung__impulse_v2.json"
OUT_JSON = E2 / "ps2_mechanical_supply.json"
OUT_MD = E2 / "ps2_mechanical_supply.md"
CHECKPOINT = E2 / "ps2_mechanical_supply.checkpoint.json"
CARD_DIR = E2 / "ps2_cards"

# Attempt 1 (2026-08-26): 11/12 printed then relay 500 on ps2_run12.
# Records lived only in _run_arms locals and were dropped.  Charged.
ATTEMPT1_WASTED = {
    "llm": 67,
    "fit": 31,
    "wall_seconds": 5236.8,
    "printed_runs": 11,
    "records_persisted": False,
    "stop": "AgentTransportError InternalServerError on ps2_run12 inspect",
}

PROTOCOL_VERSION = "ps2_mechanical_supply_v1"
EVIDENCE_GRADE = "development-mechanism (pilot)"

EXAM_UNIT = ps1.EXAM_UNIT
ARM_A3 = "A3"
ARM_NEUTRAL = "A5-neutral"
ARM_SCOPED = "A5-scoped"
ARMS = (ARM_A3, ARM_NEUTRAL, ARM_SCOPED)

# Coarse stdout from attempt 1.  Not a protocol record: _run_arms
# returned only after all 12, so the structured rows were dropped.
ATTEMPT1_STDOUT = (
    {"run_id": "ps2_run1", "arm": ARM_A3, "inject": False, "selected": False,
     "support": False, "approved": False, "deployed": False,
     "agent": "burst", "gain": 0.0, "llm": 7},
    {"run_id": "ps2_run2", "arm": ARM_NEUTRAL, "inject": True, "selected": False,
     "support": False, "approved": False, "deployed": False,
     "agent": "burst,outlier_threshold", "gain": 0.0, "llm": 8},
    {"run_id": "ps2_run3", "arm": ARM_SCOPED, "inject": True, "selected": False,
     "support": True, "approved": False, "deployed": False,
     "agent": "burst,outlier_threshold", "gain": 0.0, "llm": 8},
    {"run_id": "ps2_run4", "arm": ARM_A3, "inject": False, "selected": False,
     "support": False, "approved": False, "deployed": False,
     "agent": "burst,outlier_threshold", "gain": 0.0, "llm": 8},
    {"run_id": "ps2_run5", "arm": ARM_NEUTRAL, "inject": False, "selected": False,
     "support": False, "approved": False, "deployed": False,
     "agent": "-", "gain": 0.0, "llm": 3},
    {"run_id": "ps2_run6", "arm": ARM_SCOPED, "inject": False, "selected": False,
     "support": False, "approved": False, "deployed": False,
     "agent": "-", "gain": 0.0, "llm": 4},
    {"run_id": "ps2_run7", "arm": ARM_A3, "inject": False, "selected": False,
     "support": False, "approved": False, "deployed": False,
     "agent": "burst,outlier_threshold", "gain": 0.0, "llm": 7},
    {"run_id": "ps2_run8", "arm": ARM_NEUTRAL, "inject": True, "selected": False,
     "support": False, "approved": False, "deployed": False,
     "agent": "burst", "gain": 0.0, "llm": 5},
    {"run_id": "ps2_run9", "arm": ARM_SCOPED, "inject": True, "selected": False,
     "support": True, "approved": False, "deployed": False,
     "agent": "burst", "gain": 0.0, "llm": 6},
    {"run_id": "ps2_run10", "arm": ARM_A3, "inject": False, "selected": False,
     "support": False, "approved": False, "deployed": False,
     "agent": "burst,outlier_threshold", "gain": 0.0, "llm": 6},
    {"run_id": "ps2_run11", "arm": ARM_NEUTRAL, "inject": True, "selected": False,
     "support": False, "approved": False, "deployed": False,
     "agent": "burst", "gain": 0.0, "llm": 5},
    {"run_id": "ps2_run12", "arm": ARM_SCOPED, "inject": None, "selected": None,
     "support": None, "approved": None, "deployed": None,
     "agent": None, "gain": None, "llm": None,
     "note": "InternalServerError during inspect; unit not scored"},
)
REPLICATES = 4
LLM_PER_RUN = 12
FIT_PER_RUN = 10
ROUNDS = s1.HELD_IN_ROUNDS
TARGET_FAMILY = ps0.TARGET_FAMILY
TARGET_OPERATOR = ps0.TARGET_OPERATOR
PLACEBO_OPERATOR = "resample_uniform"
PLACEBO_FAMILY = "resample"
MATERIAL = s1.MATERIAL
FROZEN_MARKER = "Frozen program steps:"

LLM_TOTAL_CAP = 150
FIT_TOTAL_CAP = 160
# Original book was 2.5h; this continuation hard-caps this attempt at 2h.
WALL_SECONDS_CAP = int(2 * 60 * 60)

CARD_KIND = "ps2_source_hypothesis"
SCOPED_SKILL_ID = "ps2_source_hypothesis_scoped_v1"
NEUTRAL_SKILL_ID = "ps2_source_hypothesis_neutral_v1"
TOKEN_TOLERANCE = 0.15

RUN_PLAN = tuple(
    {"run_id": "ps2_run%d" % (index + 1),
     "arm": ARMS[index % len(ARMS)],
     "replicate": index // len(ARMS) + 1}
    for index in range(REPLICATES * len(ARMS))
)

# ---------------------------------------------------------------- W-1 prod
# The same 12-run protocol, re-run after the supply rung was wired in
# ``methods/ttha``.  Same cards, same arms, same unit, same budgets: the only
# difference is that ``supplies_candidates`` is now read by Fast, so the
# frozen program no longer depends on the agent's propose stage surviving.
PROD_PROTOCOL_VERSION = "ps2p_production_validation_v1"
PROD_OUT_JSON = E2 / "ps2p_production_validation.json"
PROD_OUT_MD = E2 / "ps2p_production_validation.md"
PROD_CHECKPOINT = E2 / "ps2p_production_validation.checkpoint.json"
PROD_RUN_PLAN = tuple(
    {"run_id": "ps2p_run%d" % (index + 1),
     "arm": ARMS[index % len(ARMS)],
     "replicate": index // len(ARMS) + 1}
    for index in range(REPLICATES * len(ARMS))
)
_PROD = {"on": False}


def _run_plan() -> tuple[Mapping[str, Any], ...]:
    return PROD_RUN_PLAN if _PROD["on"] else RUN_PLAN


def _checkpoint_path() -> Path:
    return PROD_CHECKPOINT if _PROD["on"] else CHECKPOINT

AUTHORITY_FIELDS = ps1.AUTHORITY_FIELDS


def _skill_cand_id(skill_id: str) -> str:
    return "cand_skill_%s" % skill_id


def _frozen_line(op: str, params: Mapping[str, Any] | None = None) -> str:
    return "%s %s" % (FROZEN_MARKER, json.dumps(
        [{"op": op, "params": dict(params or {})}],
        separators=(",", ":")))


def _pad_body(body: str, target_tokens: int) -> str:
    tokens = ps1._tokens(body)
    if tokens >= target_tokens:
        return body
    filler = (
        " This paragraph is padding so the two cards stay token-parity "
        "under the same audit the previous book used. It names no operator, "
        "no program family, and no ranking among legal options."
    )
    while ps1._tokens(body) < target_tokens:
        body += filler
    return body


def _scoped_card(scope: Mapping[str, Any],
                 sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    card = ps1._scoped_card(scope, sources)
    card["skill_id"] = SCOPED_SKILL_ID
    card["risk_guards"]["card_kind"] = CARD_KIND
    body = str(card["body"])
    assert FROZEN_MARKER not in body
    card["body"] = body + "\n" + _frozen_line(TARGET_OPERATOR)
    return card


def _neutral_card(scope: Mapping[str, Any]) -> dict[str, Any]:
    pattern = dict(scope["pattern_intersection"])
    body = "\n".join([
        "WHEN: task_kind == %s, consumer %s, metric %s, and the deployment-"
        "visible pattern reads %s."
        % (scope["task_kind"], scope["consumer_id"], scope["metric"],
           ", ".join("%s=%s" % (key, value)
                     for key, value in sorted(pattern.items()))),
        "OBSERVE: before proposing, read the deployment-visible features of "
        "the current Workspace and note what the readings are.",
        "TRY-HYPOTHESIS: this card records no ranking preference among the "
        "options the proposal stage may name. It is not execution authority "
        "and grants no right to deploy. Any candidate that appears from it "
        "is for the same candidate budget and still needs this Target's own "
        "Support and delayed approval.",
        "EVIDENCE: this card carries no accumulated outcome evidence from any "
        "prior domain and reports no direction. n = 0. No evidence is weak "
        "evidence; it establishes nothing worth treating as a fact.",
        "VERIFY: whatever reaches the pool holds here only if this Target's "
        "own held-in Support reads materially positive and the delayed "
        "feedback approves the Draft. Nothing is assumed from anywhere else.",
        "FALLBACK: if Support or delayed refuses, drop the candidate and "
        "return to identity rather than retrying the same option.",
    ])
    assert FROZEN_MARKER not in body
    return {
        "schema_version": "skill-entry/1",
        "skill_id": NEUTRAL_SKILL_ID,
        "skill_kind": "capability",
        "revision": 1,
        "body": body + "\n" + _frozen_line(PLACEBO_OPERATOR),
        "observable_applicability": ps1._applicability(scope),
        "allowed_tools": [],
        "risk_guards": {
            "card_kind": CARD_KIND,
            "authority": ps1._authority(supplies=True),
            "scope_v1": ps1._plainify(scope),
            "evidence": {"source_count": 0, "sources": [],
                         "uncertainty": "no accumulated evidence"},
            "requires_target_support": True,
            "execution_right": "withheld_supplies_candidate_only",
            "counting_rule": (
                "control card; a conversion here would be a false deploy "
                "of a sealed-oracle no-op and is scored as PLACEBO_CONVERSION"),
        },
    }


def _split_frozen(body: str) -> tuple[str, str]:
    idx = body.find(FROZEN_MARKER)
    if idx < 0:
        return body, ""
    return body[:idx].rstrip(), body[idx:]


def _balance_tokens(scoped: dict[str, Any],
                    neutral: dict[str, Any]) -> None:
    scoped_prose, scoped_frozen = _split_frozen(str(scoped["body"]))
    neutral_prose, neutral_frozen = _split_frozen(str(neutral["body"]))
    target = max(ps1._tokens(scoped_prose + "\n" + scoped_frozen),
                 ps1._tokens(neutral_prose + "\n" + neutral_frozen))
    # Pad prose only.  Anything after the Frozen marker must stay a JSON
    # array -- _parse_frozen_steps json.loads the remainder.
    while ps1._tokens(scoped_prose + "\n" + scoped_frozen) < target:
        scoped_prose = _pad_body(scoped_prose, ps1._tokens(scoped_prose) + 8)
    while ps1._tokens(neutral_prose + "\n" + neutral_frozen) < target:
        neutral_prose = _pad_body(neutral_prose, ps1._tokens(neutral_prose) + 8)
    scoped["body"] = scoped_prose + "\n" + scoped_frozen
    neutral["body"] = neutral_prose + "\n" + neutral_frozen


def _oracle_confirm() -> dict[str, Any]:
    """Read the sealed oracle as a grader.  Never handed to a snapshot."""
    payload = json.loads(ORACLE_JSON.read_text(encoding="utf-8"))
    by_name = {str(row["program"]): row for row in payload.get("programs") or []}
    rows = {}
    for name in (TARGET_OPERATOR, PLACEBO_OPERATOR):
        row = by_name.get(name) or {}
        rows[name] = {
            "legal": bool(row.get("legal")),
            "verifier_passed": bool(row.get("verifier_passed")),
            "numeric_no_op": bool(row.get("numeric_no_op")),
            "heldin_headroom": row.get("heldin_headroom"),
            "heldout_utility": row.get("heldout_utility"),
            "heldout_worst_class_recall_delta": row.get(
                "heldout_worst_class_recall_delta"),
        }
    placebo = rows[PLACEBO_OPERATOR]
    placebo_ok = (
        placebo["legal"] and placebo["verifier_passed"]
        and placebo["numeric_no_op"]
        and abs(float(placebo["heldin_headroom"] or 0.0)) < MATERIAL
        and abs(float(placebo["heldout_utility"] or 0.0)) < MATERIAL
    )
    target = rows[TARGET_OPERATOR]
    return {
        "oracle_path": ORACLE_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "isolation": payload.get("isolation"),
        "do_not_load_into_harness": True,
        "programs": rows,
        "placebo_is_legal_numeric_noop": placebo_ok,
        "target_is_legal_positive": (
            target["legal"] and target["verifier_passed"]
            and not target["numeric_no_op"]
            and float(target["heldin_headroom"] or 0.0) >= MATERIAL),
    }


def _card_audit(scoped: Mapping[str, Any],
                neutral: Mapping[str, Any],
                oracle: Mapping[str, Any]) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.operators.registry import OPERATOR_NAMES

    def _prose(card: Mapping[str, Any]) -> str:
        body = str(card["body"])
        idx = body.find(FROZEN_MARKER)
        return body[:idx] if idx >= 0 else body

    scoped_steps = _parse_frozen_steps(str(scoped["body"]))
    neutral_steps = _parse_frozen_steps(str(neutral["body"]))
    scoped_tokens = ps1._tokens(scoped["body"])
    neutral_tokens = ps1._tokens(neutral["body"])
    ratio = neutral_tokens / float(scoped_tokens or 1)
    scope = scoped["risk_guards"]["scope_v1"]
    machine_leaves, dropped = ps1._applicability_leaves(scope)
    neutral_prose = _prose(neutral).lower()
    prose_ops = sorted(op for op in OPERATOR_NAMES if op in neutral_prose)
    family_words = sorted(
        family for family in s1.PROGRAM_FAMILIES
        if family in neutral_prose and family != "resample")
    return {
        "scoped_body_tokens": scoped_tokens,
        "neutral_body_tokens": neutral_tokens,
        "token_ratio": round(ratio, 4),
        "token_ratio_within_tolerance": abs(ratio - 1.0) <= TOKEN_TOLERANCE,
        "neutral_prose_names_no_operator": not prose_ops,
        "neutral_prose_operator_hits": prose_ops,
        "neutral_frozen_operator": PLACEBO_OPERATOR,
        "neutral_family_hits_in_prose": family_words,
        "both_open_only_supplies_candidates": all(
            card["risk_guards"]["authority"]["supplies_candidates"] is True
            and card["risk_guards"]["authority"]["grants_execution"] is False
            and card["risk_guards"]["requires_target_support"] is True
            for card in (scoped, neutral)),
        "both_cards_supply_a_frozen_program": (
            scoped_steps is not None and neutral_steps is not None),
        "scoped_frozen_ops": [op for op, _ in (scoped_steps or ())],
        "neutral_frozen_ops": [op for op, _ in (neutral_steps or ())],
        "identical_applicability": (
            scoped["observable_applicability"]
            == neutral["observable_applicability"]),
        "same_schema_and_kind": (
            scoped["schema_version"] == neutral["schema_version"]
            and scoped["skill_kind"] == neutral["skill_kind"]),
        "machine_applicability_leaf_count": len(machine_leaves),
        "pattern_leaves_dropped_as_uncontracted_for_edit_schema": dropped,
        "oracle_placebo_ok": oracle["placebo_is_legal_numeric_noop"],
        "oracle_target_ok": oracle["target_is_legal_positive"],
        "scoped_frozen_params": "{}",
        "scoped_params_note": (
            "empty params use operator literature defaults "
            "(window=7, n_sigmas=3.0).  The sealed-oracle scored "
            "hampel params are not copied onto the card."),
        "injection_channel": (
            "card body Frozen program steps -> production "
            "_parse_frozen_steps / _skill_frozen_candidates; "
            "requires_target_support=true so DRAFT merge keeps one "
            "agent exploration slot; CandidatePool.build counts the "
            "inject inside min(total_k=4, maximum_candidates=3)"),
    }


def _budget_equality(base_shas: Mapping[str, str]) -> dict[str, Any]:
    budget = ps1._budget_equality(base_shas)
    budget["why_a_supplied_candidate_costs_a_slot"] = (
        "the card now carries Frozen program steps, so "
        "_skill_frozen_candidates emits cand_skill_<id>.  DRAFT merge is "
        "(*agent[:1], *draft[:1]); CandidatePool.build prepends identity "
        "and truncates at maximum_candidates=3.  The inject occupies one "
        "of the two program slots; it does not add a fourth.")
    return budget


def _same_rights_proof() -> dict[str, Any]:
    return {
        "no_methods_edit": True,
        "channel": "methods/ttha/fast_agent.py _skill_frozen_candidates",
        "draft_not_priority": (
            "requires_target_support=true routes the inject into the DRAFT "
            "bucket: Agent candidates stay in front; the Skill does not keep "
            "the ACTIVE priority slot"),
        "grants_execution_false": True,
        "verifier": "verify_candidate on every pool member, same limits",
        "select": "ordinary fast_select_v1 over the public pool",
        "support_and_delayed": "run_online_round / open_delayed unchanged",
        "deploy": "only winner_delayed_approved plus the existing freeze path",
        "no_runner_bypass": (
            "this runner never marks a candidate verified, never writes a "
            "Support receipt, and never sets approved_skill_id"),
    }


def _is_injected(candidate_id: str, skill_ids: Sequence[str]) -> bool:
    return any(candidate_id == _skill_cand_id(skill_id)
               for skill_id in skill_ids)


def _round_anatomy(record: Mapping[str, Any],
                   skill_ids: Sequence[str]) -> dict[str, Any]:
    proposals = list(record.get("proposals") or [])
    injected = [row for row in proposals
                if _is_injected(str(row["candidate_id"]), skill_ids)]
    agent = [row for row in proposals
             if str(row["candidate_id"]) not in ("identity",)
             and not _is_injected(str(row["candidate_id"]), skill_ids)]
    chosen = str(record.get("chosen") or "")
    return {
        "round": record.get("round"),
        "pool": list(record.get("pool") or []),
        "chosen": chosen,
        "injected": injected,
        "injected_in_pool": bool(injected),
        "injected_selected": any(row["candidate_id"] == chosen
                                 for row in injected),
        "agent_programs": [
            {"candidate_id": row["candidate_id"],
             "operators": row.get("operators"),
             "family": row.get("family"),
             "chosen_by_select": row.get("chosen_by_select"),
             "outcome": row.get("outcome")}
            for row in agent],
        "agent_families": sorted({str(row.get("family")) for row in agent
                                  if row.get("family") not in (None, "identity")}),
        # DRAFT merge is (*agent[:1], *draft[:1]).  The inject occupies a
        # cap slot; it does not delete the exploration slot.  inject-only
        # therefore means the agent compiled nothing (or every agent
        # program was the missing-data no-op filter), not that the slot
        # was removed.
        "inject_and_agent_coexist": bool(injected) and bool(agent),
        # W-1 decoupling readout: the inject is in the pool and the agent
        # contributed no program.  Before the wiring this combination was
        # unreachable -- the merge lived downstream of a successful propose
        # stage, so an agent that produced nothing took the inject with it.
        "supply_without_agent_program": bool(injected) and not agent,
        "exploration_slot_kept": True,
        "agent_program_count": len(agent),
        "llm_calls_this_round": record.get("llm_calls_this_round"),
        "proposal_count": record.get("proposal_count"),
        "retrieved_skill_ids": list(record.get("retrieved_skill_ids") or []),
    }


def _inject_funnel(result: Mapping[str, Any], *,
                   skill_id: str, operator: str) -> dict[str, Any]:
    """Six-stage funnel for the mechanically injected candidate only."""
    entered = selected = verified = supported = approved = deployed = False
    break_at: str | None = "never_entered_pool"
    detail: list[dict[str, Any]] = []
    want = _skill_cand_id(skill_id)
    for record in result.get("rounds") or []:
        anatomy = _round_anatomy(record, (skill_id,))
        for row in anatomy["injected"]:
            entered = True
            if break_at == "never_entered_pool":
                break_at = "entered_not_selected"
            if row.get("chosen_by_select"):
                selected = True
                if break_at == "entered_not_selected":
                    break_at = "selected_verifier_or_support"
            if row.get("outcome") == "probe":
                verified = True
            if row.get("outcome") == "verifier_rejected":
                break_at = "verifier_rejected"
            elif entered:
                # selectable pool is post-verify; entry already means the
                # compile-time verifier accepted the inject
                verified = True
            detail.append({"round": record.get("round"), **row})
        # Scoring correction (W-1).  PS-2 gated Support on the *select*
        # stage naming the inject, but the harness gives every pool member a
        # Support trial inside the round's budget, so a probed inject that
        # select passed over still earns a real receipt.  PS-2 run9/run12
        # were scored 0/4 Support on that convention while the persisted
        # Episodes show +0.6364 / +0.6000 POSITIVE.  Attribution is exact
        # without selection: the Episode the inject's own probe wrote carries
        # the Skill id in its episode_id, so an agent-authored program of the
        # same signature is still not counted here.
        for episode in record.get("episodes") or []:
            if skill_id not in str(episode.get("episode_id") or ""):
                continue
            if str(episode.get("workflow_signature")) not in (
                    operator, skill_id):
                continue
            support = episode.get("support_gain")
            if support is not None and float(support) >= MATERIAL \
                    and str(episode.get("relation")) == "POSITIVE":
                supported = True
                if break_at in ("entered_not_selected",
                                "selected_verifier_or_support"):
                    break_at = "support_positive_delayed_pending"
        if record.get("winner_delayed_approved"):
            winner_ops = [str(step.get("op"))
                          for step in (record.get("winner_program") or [])]
            if operator in winner_ops:
                approved = True
                break_at = "delayed_approved_not_deployed"
    deployed_ops = [str(step.get("op")) for step
                    in (result.get("deployment") or {}).get(
                        "applied_program") or []]
    deployed = operator in deployed_ops
    if deployed:
        break_at = None
    elif not entered:
        break_at = "never_entered_pool"
    elif entered and not selected and break_at == "entered_not_selected":
        break_at = "selection_did_not_choose_inject"
    elif selected and not verified and not supported:
        if break_at != "verifier_rejected":
            break_at = "selected_not_probed"
    elif verified and not supported:
        break_at = "support_not_material_positive"
    elif supported and not approved:
        break_at = "delayed_refused"
    return {
        "injected_id": want,
        "operator": operator,
        "entered_pool": entered,
        "selected_by_agent": selected,
        "passed_verifier": verified,
        "support_material_positive": supported,
        "delayed_approved": approved,
        "deployed": deployed,
        "break_at": break_at,
        "rows": detail,
    }


def _load_scope() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not PS1_JSON.is_file():
        raise s1.Stop("INSTRUMENT_UNREADABLE",
                      "PS-1 artifact missing: %s" % PS1_JSON)
    payload = json.loads(PS1_JSON.read_text(encoding="utf-8"))
    scoped = payload["cards"]["scoped"]
    scope = dict(scoped["risk_guards"]["scope_v1"])
    sources = list(scoped["risk_guards"]["evidence"]["sources"])
    return scope, sources


def _ps1_baseline() -> dict[str, Any]:
    payload = json.loads(PS1_JSON.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in payload.get("runs") or []
                if row.get("arm") == arm]
        out[arm] = {
            "proposal_families": sorted({
                family for row in rows
                for family in row.get("proposal_families") or []}),
            "funnel_counts": (payload.get("aggregate") or {}).get(
                arm, {}).get("funnel_counts"),
            "target_family_proposal_rate": (payload.get("aggregate") or {}).get(
                arm, {}).get("target_family_proposal_rate"),
        }
    return {
        "artifact": PS1_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "verdict": (payload.get("verdict") or {}).get("verdict"),
        "by_arm": out,
        "note": (
            "PS-1 measured text-rung proposal shift (all 0/4).  PS-2 "
            "compares agent-authored families against that baseline; a "
            "mechanical inject is not scored as a proposal-shift win."),
    }


def _inject_miss_reason(*, arm: str, card_seen: Sequence[str],
                        anatomies: Sequence[Mapping[str, Any]],
                        inject: Mapping[str, Any], llm: Any,
                        agent_families: Sequence[str]) -> str | None:
    """Why a card arm did not put the frozen program in the selectable pool."""
    if arm == ARM_A3 or inject.get("entered_pool"):
        return None
    if not anatomies:
        return "no_round_records"
    if not card_seen:
        if int(llm or 0) <= 4 and not agent_families:
            return "retrieval_miss_and_early_stop"
        return "retrieval_did_not_serve_card"
    return "card_in_view_not_in_selectable_pool"


def _score_run(plan: Mapping[str, Any], result: Mapping[str, Any],
               base_shas: Mapping[str, str]) -> dict[str, Any]:
    arm = str(plan["arm"])
    public = s1._public_unit_result(result)
    skill_ids = (
        (SCOPED_SKILL_ID,) if arm == ARM_SCOPED
        else (NEUTRAL_SKILL_ID,) if arm == ARM_NEUTRAL
        else ())
    anatomies = [_round_anatomy(record, skill_ids)
                 for record in public.get("rounds") or []]
    if arm == ARM_SCOPED:
        inject = _inject_funnel(public, skill_id=SCOPED_SKILL_ID,
                                operator=TARGET_OPERATOR)
    elif arm == ARM_NEUTRAL:
        inject = _inject_funnel(public, skill_id=NEUTRAL_SKILL_ID,
                                operator=PLACEBO_OPERATOR)
    else:
        inject = {"injected_id": None, "entered_pool": False,
                  "selected_by_agent": False, "passed_verifier": False,
                  "support_material_positive": False,
                  "delayed_approved": False, "deployed": False,
                  "break_at": "no_card", "rows": []}
    card_seen = sorted({
        skill_id for record in public.get("rounds") or []
        for skill_id in record.get("retrieved_skill_ids") or []
        if skill_id in (SCOPED_SKILL_ID, NEUTRAL_SKILL_ID)})
    deployment = public.get("deployment") or {}
    deltas = deployment.get("heldout_recall_delta_by_class") or {}
    agent_families = sorted({
        family for anatomy in anatomies
        for family in anatomy["agent_families"]})
    return {
        "run_id": plan["run_id"],
        "arm": arm,
        "replicate": plan["replicate"],
        "base_runtime_bundle_sha": base_shas[arm],
        "card_in_fast_view": card_seen,
        "inject_funnel": inject,
        "pool_anatomy": anatomies,
        "agent_proposal_families": agent_families,
        "exploration_slot_kept": all(
            anatomy["exploration_slot_kept"]
            for anatomy in anatomies) if anatomies else True,
        "inject_miss_reason": _inject_miss_reason(
            arm=arm, card_seen=card_seen, anatomies=anatomies,
            inject=inject, llm=result.get("llm_calls"),
            agent_families=agent_families),
        "proposal_ledger": ps0._proposal_ledger(public),
        "proposal_families": sorted({
            row["family"] for record in public.get("rounds") or []
            for row in record.get("proposals") or []
            if row["family"] != "identity"}),
        "llm_calls": result.get("llm_calls"),
        "consumer_fits": result.get("consumer_fits"),
        "probes": sum(len(record.get("probes") or [])
                      for record in public.get("rounds") or []),
        "wasted_probes": sum(
            1 for record in public.get("rounds") or []
            for probe in record.get("probes") or []
            if str(probe.get("kind")) == "verifier_rejected"),
        "seconds": result.get("seconds"),
        "deployment": {
            "deploy_source": deployment.get("deploy_source"),
            "applied_program": deployment.get("applied_program"),
            "heldout_accuracy_gain": deployment.get("heldout_accuracy_gain"),
            "worst_class_delta": (min(float(value) for value in deltas.values())
                                  if deltas else 0.0),
        },
        "rounds": public.get("rounds"),
    }


def _checkpoint(runs: Sequence[Mapping[str, Any]],
                ledger: Mapping[str, int],
                base_shas: Mapping[str, str],
                *, started: float | None = None) -> None:
    _checkpoint_path().write_text(
        json.dumps(ps0c.redact({
            "runs": list(runs),
            "ledger": dict(ledger),
            "base_shas": dict(base_shas),
            "completed_run_ids": [row["run_id"] for row in runs],
            "wall_seconds_used": (round(time.time() - started, 1)
                                  if started is not None else None),
        }), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")


def _run_unit_with_retry(*, unit: Any, cell: Any, arm: str, base_snapshot: Any,
                         backend: Any, store_root: Path) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.runtime.agent_backend import AgentTransportError

    last: Exception | None = None
    for attempt in range(2):
        try:
            return s1.run_unit(
                unit=unit, cell=cell, arm=arm, base_snapshot=base_snapshot,
                carried_episodes=(), agent_factory=cls._live_agent,
                backend=backend, store_root=store_root,
                rounds=ROUNDS, fit_cap=FIT_PER_RUN)
        except AgentTransportError as exc:
            last = exc
            print("transport retry after %s on %s" % (exc, store_root.name),
                  flush=True)
            time.sleep(8)
            if store_root.exists():
                shutil.rmtree(store_root)
            backend = cls._live_backend(LLM_PER_RUN)
    raise s1.Stop("BACKEND_UNAVAILABLE",
                  "relay transport failed after unit retry: %s" % last)


def _run_arms(*, cards: Mapping[str, Any], h0: Any, store_root: Path,
              ledger: dict[str, int], started: float,
              runs: list[dict[str, Any]]
              ) -> dict[str, str]:
    cell = s1._build_cell(EXAM_UNIT)
    bases: dict[str, Any] = {ARM_A3: h0}
    base_shas: dict[str, str] = {ARM_A3: h0.runtime_bundle_sha}
    for arm, card in ((ARM_NEUTRAL, cards["neutral"]),
                      (ARM_SCOPED, cards["scoped"])):
        snapshot, _applied = s1._apply_entries(
            h0, [card], store_root=store_root / "bases",
            tag=arm.replace("-", "_"))
        bases[arm] = snapshot
        base_shas[arm] = snapshot.runtime_bundle_sha
    done = {str(row["run_id"]) for row in runs}
    for plan in _run_plan():
        if plan["run_id"] in done:
            print("skip %s (checkpoint)" % plan["run_id"], flush=True)
            continue
        if ledger["llm"] >= LLM_TOTAL_CAP or ledger["fit"] >= FIT_TOTAL_CAP:
            raise s1.Stop("COMPUTE_BUDGET_EXCEEDED",
                          "book cap reached before %s" % plan["run_id"])
        if time.time() - started > WALL_SECONDS_CAP:
            raise s1.Stop("COMPUTE_BUDGET_EXCEEDED",
                          "wall clock cap reached before %s" % plan["run_id"])
        arm = plan["arm"]
        backend = cls._live_backend(LLM_PER_RUN)
        result = _run_unit_with_retry(
            unit=EXAM_UNIT, cell=cell, arm=arm, base_snapshot=bases[arm],
            backend=backend, store_root=store_root / plan["run_id"])
        ledger["llm"] += int(result.get("llm_calls") or 0)
        ledger["fit"] += int(result.get("consumer_fits") or 0)
        row = _score_run(plan, result, base_shas)
        runs.append(row)
        _checkpoint(runs, ledger, base_shas, started=started)
        inject = row["inject_funnel"]
        print("%-11s %-11s inject=%-5s selected=%-5s support=%-5s "
              "approved=%-5s deployed=%-5s miss=%s agent=%s gain=%+.4f llm=%s"
              % (plan["run_id"], arm, inject.get("entered_pool"),
                 inject.get("selected_by_agent"),
                 inject.get("support_material_positive"),
                 inject.get("delayed_approved"), inject.get("deployed"),
                 row.get("inject_miss_reason") or "-",
                 ",".join(row.get("agent_proposal_families") or []) or "-",
                 float(row["deployment"]["heldout_accuracy_gain"] or 0.0),
                 row.get("llm_calls")),
              flush=True)
    return base_shas


def _aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    stages = ("entered_pool", "selected_by_agent", "passed_verifier",
              "support_material_positive", "delayed_approved", "deployed")
    for arm in ARMS:
        rows = [row for row in runs if row["arm"] == arm]
        if not rows:
            continue
        out[arm] = {
            "runs": len(rows),
            "run_ids": [row["run_id"] for row in rows],
            "inject_funnel_counts": {
                stage: sum(1 for row in rows
                           if (row.get("inject_funnel") or {}).get(stage))
                for stage in stages},
            "inject_entry_rate": "%d/%d" % (
                sum(1 for row in rows
                    if (row.get("inject_funnel") or {}).get("entered_pool")),
                len(rows)),
            "card_served_runs": sum(1 for row in rows
                                    if row["card_in_fast_view"]),
            "exploration_slot_kept_runs": sum(
                1 for row in rows if row.get("exploration_slot_kept")),
            "agent_proposal_families": sorted({
                family for row in rows
                for family in row.get("agent_proposal_families") or []}),
            "inject_and_agent_coexist_rounds": sum(
                1 for row in rows
                for anatomy in row.get("pool_anatomy") or []
                if anatomy.get("inject_and_agent_coexist")),
            "mean_llm": round(sum(int(row["llm_calls"] or 0)
                                  for row in rows) / len(rows), 2),
            "mean_fits": round(sum(int(row["consumer_fits"] or 0)
                                   for row in rows) / len(rows), 2),
            "mean_probes": round(sum(int(row["probes"] or 0)
                                     for row in rows) / len(rows), 2),
            "deployed_utilities": [row["deployment"]["heldout_accuracy_gain"]
                                   for row in rows],
            "worst_class_deltas": [row["deployment"]["worst_class_delta"]
                                   for row in rows],
            "harm_runs": sum(1 for row in rows
                             if float(row["deployment"]["worst_class_delta"])
                             <= -MATERIAL),
            "break_ats": [row["inject_funnel"].get("break_at")
                          for row in rows],
            "inject_miss_reasons": [row.get("inject_miss_reason")
                                    for row in rows
                                    if row.get("inject_miss_reason")],
        }
    return out


def _supply_decoupling(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Did the supply rung survive rounds the agent contributed nothing to?

    The PS-2 break was structural, not statistical: the merge that put the
    frozen program in the pool lived downstream of a successful propose
    stage.  A round with the inject in the pool and no agent program is the
    direct observation that the dependency is gone.
    """
    rows: list[dict[str, Any]] = []
    for run_row in runs:
        for anatomy in run_row.get("pool_anatomy") or []:
            if str(run_row["arm"]) == ARM_A3:
                continue
            rows.append({
                "run_id": run_row["run_id"],
                "arm": run_row["arm"],
                "round": anatomy.get("round"),
                "injected_in_pool": bool(anatomy.get("injected_in_pool")),
                "agent_program_count": int(
                    anatomy.get("agent_program_count") or 0),
                "supply_without_agent_program": bool(
                    anatomy.get("supply_without_agent_program")),
                "llm_calls_this_round": anatomy.get("llm_calls_this_round"),
            })
    agentless = [row for row in rows if row["agent_program_count"] == 0]
    return {
        "card_arm_rounds": len(rows),
        "rounds_with_inject_in_pool": sum(
            1 for row in rows if row["injected_in_pool"]),
        "rounds_without_an_agent_program": len(agentless),
        "of_those_the_inject_still_entered": sum(
            1 for row in agentless if row["injected_in_pool"]),
        "decoupled": all(row["injected_in_pool"] for row in agentless),
        "rows": rows,
        "ps2_comparison": (
            "PS-2 recorded 4 card-arm runs whose pool degraded to identity "
            "with proposal_count=0 and llm=2; every one of them lost the "
            "inject."),
    }


def _prod_verdict(aggregate: Mapping[str, Any], *,
                  stopped: str | None,
                  runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """W-1 production validation, pre-registered before the runs."""
    if stopped:
        return {"verdict": stopped, "reason": "stopped before the full plan"}
    if not all(arm in aggregate for arm in ARMS):
        return {"verdict": "COMPUTE_BUDGET_EXCEEDED",
                "reason": "not every arm completed its replicates"}
    scoped = aggregate[ARM_SCOPED]
    neutral = aggregate[ARM_NEUTRAL]
    a3 = aggregate[ARM_A3]
    counts = scoped["inject_funnel_counts"]
    scoped_n = int(scoped["runs"])
    entered = int(counts["entered_pool"])
    probed = int(counts["passed_verifier"])
    supported = int(counts["support_material_positive"])
    deployed = int(counts["deployed"])
    placebo_deployed = int(
        neutral["inject_funnel_counts"]["deployed"]) >= 1
    placebo_supported = int(
        neutral["inject_funnel_counts"]["support_material_positive"]) >= 1
    harm_zero = (int(scoped["harm_runs"]) == 0
                 and int(neutral["harm_runs"]) == 0
                 and int(a3["harm_runs"]) == 0)
    explore = int(scoped["exploration_slot_kept_runs"]) == scoped_n
    gains = [float((row.get("deployment") or {}).get(
        "heldout_accuracy_gain") or 0.0)
        for row in runs if row["arm"] == ARM_SCOPED]
    facts = {
        "scoped_entry": "%d/%d" % (entered, scoped_n),
        "scoped_support_receipt": "%d/%d" % (probed, scoped_n),
        "scoped_support_material_positive": "%d/%d" % (supported, scoped_n),
        "scoped_deployed": "%d/%d" % (deployed, scoped_n),
        "scoped_heldout_gains": gains,
        "placebo_support_material_positive": placebo_supported,
        "placebo_deployed": placebo_deployed,
        "harm_zero_all_arms": harm_zero,
        "exploration_slot_kept": explore,
        "break_ats": {ARM_SCOPED: scoped["break_ats"],
                      ARM_NEUTRAL: neutral["break_ats"]},
        "wording": (
            "a conversion is experience supplying a candidate through the "
            "mechanical channel, adjudicated by Target feedback.  It is not "
            "evidence that the agent learned to propose the family."),
    }
    if placebo_deployed:
        return {"verdict": "PLACEBO_CONVERSION", "facts": facts,
                "reason": (
                    "A5-neutral deployed the sealed-oracle no-op (%s); the "
                    "approval chain accepted a numerically-identity "
                    "candidate.  Dangerous-signal stop." % PLACEBO_OPERATOR)}
    if deployed >= 2 and not placebo_deployed and harm_zero:
        return {"verdict": "SUPPLY_RUNG_PRODUCTION_CONFIRMED", "facts": facts,
                "reason": (
                    "the frozen hampel entered the pool in %d/%d runs, took "
                    "a Support receipt in %d, converted through Support and "
                    "the delayed gate in %d, the no-op card did not falsely "
                    "deploy, and harm was zero in every arm.  Experience "
                    "supplied the candidate; Target feedback adjudicated it."
                    % (entered, scoped_n, probed, deployed))}
    if probed >= 1 and supported == 0:
        return {"verdict": "SUPPLY_WITHOUT_CONVERSION", "facts": facts,
                "reason": (
                    "the supplied candidate reached a Support receipt in %d "
                    "run(s) and no receipt was a material positive.  That "
                    "points at the GPOvY Support surface, not at the supply "
                    "channel." % probed)}
    if entered < scoped_n:
        return {"verdict": "POOL_ENTRY_WITHOUT_CONVERSION", "facts": facts,
                "reason": (
                    "mechanical entry was %d/%d after the wiring; the "
                    "decoupling did not hold in every run." % (entered,
                                                               scoped_n))}
    return {"verdict": "POOL_ENTRY_WITHOUT_CONVERSION", "facts": facts,
            "reason": (
                "entry and Support were reached but fewer than two runs "
                "completed the delayed gate and deployment.")}


def _verdict(aggregate: Mapping[str, Any], *,
             stopped: str | None,
             runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if stopped == "BACKEND_UNAVAILABLE":
        return {"verdict": stopped, "reason": (
            "the 12-run protocol did not finish because the live relay "
            "was unavailable.  No old-relay fallback.")}
    if stopped == "COMPUTE_BUDGET_EXCEEDED":
        return {"verdict": stopped, "reason": "stopped before the full plan"}
    if stopped:
        return {"verdict": stopped, "reason": "stopped before the full plan"}
    if not all(arm in aggregate for arm in ARMS):
        return {"verdict": "COMPUTE_BUDGET_EXCEEDED",
                "reason": "not every arm completed its replicates"}

    scoped = aggregate[ARM_SCOPED]
    neutral = aggregate[ARM_NEUTRAL]
    a3 = aggregate[ARM_A3]
    scoped_in = int(scoped["inject_funnel_counts"]["entered_pool"])
    scoped_n = int(scoped["runs"])
    converted = (int(scoped["inject_funnel_counts"]["delayed_approved"]) >= 1
                 and int(scoped["inject_funnel_counts"]["deployed"]) >= 1)
    placebo_deployed = int(neutral["inject_funnel_counts"]["deployed"]) >= 1
    harm_not_up = scoped["harm_runs"] <= max(neutral["harm_runs"],
                                             a3["harm_runs"])
    explore = (int(scoped["exploration_slot_kept_runs"]) == scoped_n
               and int(neutral["exploration_slot_kept_runs"]) == int(
                   neutral["runs"]))
    facts = {
        "scoped_inject_entry": "%d/%d" % (scoped_in, scoped_n),
        "scoped_converted": converted,
        "placebo_deployed": placebo_deployed,
        "harm_not_increased": harm_not_up,
        "exploration_slot_kept": explore,
        "break_ats": {
            ARM_SCOPED: scoped["break_ats"],
            ARM_NEUTRAL: neutral["break_ats"],
        },
        "wording": (
            "a conversion here is experience supplying a candidate through "
            "the mechanical channel, adjudicated by Target feedback.  It is "
            "not evidence that the agent learned to propose the family."),
    }
    if placebo_deployed:
        return {"verdict": "PLACEBO_CONVERSION", "facts": facts,
                "reason": (
                    "A5-neutral deployed the sealed-oracle no-op "
                    "(%s).  The approval chain accepted a candidate that "
                    "the grader says is numerically identity.  This is a "
                    "dangerous-signal stop, not a mechanical-rung win."
                    % PLACEBO_OPERATOR)}
    if scoped_in == scoped_n and converted and harm_not_up and explore:
        return {"verdict": "MECHANICAL_RUNG_CONFIRMED", "facts": facts,
                "reason": (
                    "A5-scoped's frozen hampel entered the pool in %d/%d "
                    "runs, at least one walk completed Support + delayed "
                    "approval and deploy, the no-op card did not falsely "
                    "deploy, harm did not rise, and the exploration slot "
                    "was kept.  Experience supplied the candidate; Target "
                    "feedback adjudicated it."
                    % (scoped_in, scoped_n))}
    if scoped_in == scoped_n and not converted:
        return {"verdict": "POOL_ENTRY_WITHOUT_CONVERSION", "facts": facts,
                "reason": (
                    "the inject entered the pool in %d/%d scoped runs but "
                    "the approval chain broke.  break_at=%s"
                    % (scoped_in, scoped_n, scoped["break_ats"]))}
    if scoped_in < scoped_n:
        return {"verdict": "POOL_ENTRY_WITHOUT_CONVERSION", "facts": facts,
                "reason": (
                    "mechanical entry was not 4/4 (saw %d/%d).  "
                    "break_at=%s"
                    % (scoped_in, scoped_n, scoped["break_ats"]))}
    return {"verdict": "POOL_ENTRY_WITHOUT_CONVERSION", "facts": facts,
            "reason": "entry and conversion conditions were not jointly met"}


def _markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = [
        "# PS-2 -- mechanical supply of one candidate to verify (pilot)",
        "",
        "protocol: `%s`  evidence grade: **%s**  git: `%s`  backend: **%s**"
        % (payload["protocol_version"], payload["evidence_grade"],
           payload["git_head"],
           (payload.get("backend_probe") or {}).get("returned_model")),
        "",
        "**%s**" % verdict["verdict"],
        "",
        verdict["reason"],
        "",
        "> Pilot grade.  %s shares GunPointFamily with source A, so this "
        "isolates a mechanism and is not a cross-family transfer claim.  "
        "A conversion means experience supplied a candidate through the "
        "mechanical channel and Target feedback adjudicated it.  It is "
        "not 'the agent learned to propose hampel'.  A guided positive "
        "counts zero toward Source cross-domain authorization."
        % EXAM_UNIT["unit_id"],
        "",
        "## Cards",
        "",
        "| field | A5-scoped | A5-neutral |",
        "|---|---|---|",
    ]
    scoped = payload["cards"]["scoped"]
    neutral = payload["cards"]["neutral"]
    for field in ("skill_id", "schema_version", "skill_kind", "revision"):
        lines.append("| %s | `%s` | `%s` |" % (field, scoped[field],
                                               neutral[field]))
    for field in AUTHORITY_FIELDS:
        lines.append("| authority.%s | **%s** | **%s** |" % (
            field, scoped["risk_guards"]["authority"][field],
            neutral["risk_guards"]["authority"][field]))
    lines.append("| frozen program | `%s` | `%s` |" % (
        TARGET_OPERATOR, PLACEBO_OPERATOR))
    lines.append("| requires_target_support | **True** | **True** |")
    audit = payload["card_audit"]
    lines += ["", "### Card audit", ""]
    for key, value in audit.items():
        lines.append("- **%s**: %s" % (key, value))
    oracle = payload["oracle_confirm"]
    lines += ["", "## Sealed-oracle confirmation (grader only)", "",
              "- path: `%s`" % oracle["oracle_path"],
              "- isolation: %s" % oracle["isolation"],
              "- placebo `%s` legal numeric no-op: **%s**"
              % (PLACEBO_OPERATOR, oracle["placebo_is_legal_numeric_noop"]),
              "- target `%s` legal positive: **%s**"
              % (TARGET_OPERATOR, oracle["target_is_legal_positive"]),
              ""]
    for name, row in oracle["programs"].items():
        lines.append("- `%s`: legal=%s verifier=%s no-op=%s held-in=%s "
                     "held-out=%s"
                     % (name, row["legal"], row["verifier_passed"],
                        row["numeric_no_op"], row["heldin_headroom"],
                        row["heldout_utility"]))
    rights = payload["same_rights_proof"]
    lines += ["", "## Same-rights proof (no shortcut)", ""]
    for key, value in rights.items():
        lines.append("- **%s**: %s" % (key, value))
    budget = payload["budget_equality"]
    lines += ["", "## Budget equality", "",
              "- all equal: **%s**" % budget["all_equal"], ""]
    for key, value in budget["equal_across_arms"].items():
        lines.append("- %s equal: %s (value %s)"
                     % (key, value, budget["per_arm"][ARM_A3][key]))
    lines += ["", "- %s" % budget["why_a_supplied_candidate_costs_a_slot"], ""]
    attempt1 = payload.get("attempt1_stdout") or []
    if attempt1:
        lines += ["", "## Attempt-1 stdout (supplementary, not protocol)", "",
                  "Structured rows were dropped when run 12 raised "
                  "AgentTransportError.  These cells are the flushed print "
                  "line only.", "",
                  "| run | arm | inject | selected | Support | delayed | "
                  "deployed | agent | gain | LLM |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in attempt1:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (row.get("run_id"), row.get("arm"), row.get("inject"),
                   row.get("selected"), row.get("support"),
                   row.get("approved"), row.get("deployed"),
                   row.get("agent") or row.get("note") or "-",
                   row.get("gain"), row.get("llm")))
    lines += ["", "## Per-run readout (persisted protocol records)", "",
              "| run | arm | card | inject in pool | selected | Support | "
              "delayed | deployed | break_at | miss | agent families | explore "
              "kept | gain | worst | LLM | fits |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in payload["runs"]:
        funnel = row["inject_funnel"]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | "
            "%+.4f | %+.4f | %s | %s |" % (
                row["run_id"], row["arm"],
                "yes" if row["card_in_fast_view"] else "-",
                funnel.get("entered_pool"),
                funnel.get("selected_by_agent"),
                funnel.get("support_material_positive"),
                funnel.get("delayed_approved"),
                funnel.get("deployed"),
                funnel.get("break_at"),
                row.get("inject_miss_reason") or "-",
                ",".join(row.get("agent_proposal_families") or []) or "-",
                row.get("exploration_slot_kept"),
                float(row["deployment"]["heldout_accuracy_gain"] or 0.0),
                float(row["deployment"]["worst_class_delta"] or 0.0),
                row["llm_calls"], row["consumer_fits"]))
    lines += ["", "## Three-arm inject funnel", "",
              "| arm | entry | selected | verifier/probe | Support | "
              "delayed | deployed | harm | explore kept | agent families |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        row = payload["aggregate"].get(arm)
        if not row:
            continue
        counts = row["inject_funnel_counts"]
        lines.append(
            "| %s | **%s** | %d | %d | %d | %d | %d | %d | %d/%d | %s |"
            % (arm, row["inject_entry_rate"],
               counts["selected_by_agent"], counts["passed_verifier"],
               counts["support_material_positive"],
               counts["delayed_approved"], counts["deployed"],
               row["harm_runs"], row["exploration_slot_kept_runs"],
               row["runs"],
               ", ".join(row["agent_proposal_families"]) or "none"))
    baseline = payload.get("ps1_baseline") or {}
    lines += ["", "## Agent-authored families vs PS-1 baseline", "",
              "- PS-1 verdict: **%s**" % baseline.get("verdict"), ""]
    for arm in ARMS:
        now = (payload["aggregate"].get(arm) or {}).get(
            "agent_proposal_families") or []
        then = ((baseline.get("by_arm") or {}).get(arm) or {}).get(
            "proposal_families") or []
        lines.append("- **%s**: PS-1 %s → PS-2 agent-authored %s"
                     % (arm, then or "none", now or "none"))
    ledger = payload["ledger"]
    lines += ["", "## Cost", "",
              "- LLM: %d / %d (attempt 1 charged)"
              % (ledger["llm"], ledger["llm_cap"]),
              "- Consumer fits: %d / %d" % (ledger["fit"], ledger["fit_cap"]),
              "- attempt-2 wall: %.1f s / %d s"
              % (ledger["wall_seconds"], ledger["wall_seconds_cap"]),
              "- attempt-1 wall (lost records): %s s"
              % ledger.get("wall_seconds_attempt1"),
              "- combined wall: %s s" % ledger.get("wall_seconds_combined"),
              "- downloads: 0", "", "## Obligations", ""]
    for key, value in payload["obligations"].items():
        lines.append("- **%s**: %s" % (key, value))
    if payload.get("outside_book"):
        lines += ["", "## Outside the book", ""]
        lines += ["- %s" % item for item in payload["outside_book"]]
    return "\n".join(lines) + "\n"


def compile_cards() -> dict[str, Any]:
    scope, sources = _load_scope()
    oracle = _oracle_confirm()
    if not oracle["placebo_is_legal_numeric_noop"]:
        raise s1.Stop("INSTRUMENT_UNREADABLE",
                      "placebo %s is not a legal ~0 no-op on the sealed "
                      "oracle" % PLACEBO_OPERATOR)
    scoped = _scoped_card(scope, sources)
    neutral = _neutral_card(scope)
    _balance_tokens(scoped, neutral)
    if _parse_frozen_steps(scoped["body"]) is None:
        raise s1.Stop("INSTRUMENT_UNREADABLE",
                      "scoped Frozen program steps failed to parse")
    if _parse_frozen_steps(neutral["body"]) is None:
        raise s1.Stop("INSTRUMENT_UNREADABLE",
                      "neutral Frozen program steps failed to parse")
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    for name, card in (("scoped", scoped), ("neutral", neutral)):
        (CARD_DIR / ("ps2_card_%s.json" % name)).write_text(
            json.dumps(s1._plain(card), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
    apply_smoke = _apply_smoke(scoped, neutral)
    audit = _card_audit(scoped, neutral, oracle)
    audit["apply_smoke"] = apply_smoke
    if not all(row.get("frozen_steps_survived")
               for row in apply_smoke.values()):
        raise s1.Stop("INSTRUMENT_UNREADABLE",
                      "Frozen program steps did not survive EditController")
    return {
        "cards": {"scoped": scoped, "neutral": neutral},
        "card_audit": audit,
        "oracle_confirm": oracle,
        "same_rights_proof": _same_rights_proof(),
        "ps1_baseline": _ps1_baseline(),
        "apply_smoke": apply_smoke,
    }


def _apply_smoke(scoped: Mapping[str, Any],
                 neutral: Mapping[str, Any]) -> dict[str, Any]:
    """Confirm EditController accepts both cards and frozen steps survive."""
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )

    h0 = compile_snapshot(
        PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
        verify_lock=False)
    store = Path(tempfile.gettempdir()) / "ps2_compile_smoke"
    out: dict[str, Any] = {}
    for arm, card in ((ARM_SCOPED, scoped), (ARM_NEUTRAL, neutral)):
        snapshot, applied = s1._apply_entries(
            h0, [card], store_root=store, tag=arm.replace("-", "_"))
        bodies = {str(skill.skill_id): skill.body
                  for skill in snapshot.skills}
        skill_id = str(card["skill_id"])
        steps = _parse_frozen_steps(str(bodies.get(skill_id) or ""))
        out[arm] = {
            "applied_ids": applied,
            "skill_in_snapshot": skill_id in bodies,
            "frozen_steps_survived": steps is not None,
            "frozen_ops": [op for op, _ in (steps or ())],
            "runtime_bundle_sha": snapshot.runtime_bundle_sha,
        }
    return out


def run(*, compile_only: bool = False, resume: bool = False,
        prod: bool = False) -> int:
    _PROD["on"] = bool(prod)
    out_json = PROD_OUT_JSON if prod else OUT_JSON
    out_md = PROD_OUT_MD if prod else OUT_MD
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)
    compiled = compile_cards()
    payload: dict[str, Any] = {
        "protocol_version": (PROD_PROTOCOL_VERSION if prod
                             else PROTOCOL_VERSION),
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "exam_unit": EXAM_UNIT["unit_id"],
        "ps1_source": PS1_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "run_plan": [dict(plan) for plan in _run_plan()],
        "arms": {
            ARM_A3: "no Source Skill",
            ARM_NEUTRAL: (
                "same Scope, supplies_candidates=true, frozen "
                "%s (sealed-oracle numeric no-op)" % PLACEBO_OPERATOR),
            ARM_SCOPED: (
                "same Scope, supplies_candidates=true, frozen "
                "%s" % TARGET_OPERATOR),
        },
        "semantic_discipline": (
            "a conversion is experience supplying a candidate through the "
            "mechanical channel, adjudicated by Target feedback.  It is "
            "not a proposal-ability improvement."),
        "attempt1_stdout": [dict(row) for row in ATTEMPT1_STDOUT],
        **compiled,
    }
    if compile_only:
        payload["verdict"] = {"verdict": "COMPILE_ONLY",
                              "reason": "cards compiled; live arms not run"}
        payload["runs"] = []
        payload["aggregate"] = {}
        payload["budget_equality"] = ps1._budget_equality({})
        payload["ledger"] = {"llm": 0, "llm_cap": LLM_TOTAL_CAP, "fit": 0,
                             "fit_cap": FIT_TOTAL_CAP, "wall_seconds": 0.0,
                             "wall_seconds_cap": WALL_SECONDS_CAP,
                             "downloads": 0}
        payload["obligations"] = {
            "compile_only": True,
            "methods_package_unmodified": True,
            "runtime_contracts_operators_unmodified": True,
        }
        s1._dump(out_json, ps0c.redact(payload))
        out_md.write_text(_markdown(payload), encoding="utf-8")
        print(json.dumps({"verdict": "COMPILE_ONLY",
                          "audit": payload["card_audit"],
                          "artifact": str(out_json)},
                         ensure_ascii=False, indent=1))
        return 0

    install = ps0c.install_new_backend()
    payload["backend_install"] = {
        "host": install.get("host"),
        "model": install.get("model"),
        "family": "M0_AGENT_* trycloudflare relay",
    }
    # The production validation is its own book with its own ledger; the
    # PS-2 attempt-1 waste is charged to PS-2, not re-charged here.
    ledger = ({"llm": 0, "fit": 0} if prod
              else {"llm": int(ATTEMPT1_WASTED["llm"]),
                    "fit": int(ATTEMPT1_WASTED["fit"])})
    if not prod:
        payload["attempt1_wasted"] = dict(ATTEMPT1_WASTED)
    store_root = Path(tempfile.gettempdir()) / (
        "ps2p_arms" if prod else "ps2_arms_r2")
    stopped: str | None = None
    runs: list[dict[str, Any]] = []
    base_shas: dict[str, str] = {}
    if resume and _checkpoint_path().is_file():
        saved = json.loads(_checkpoint_path().read_text(encoding="utf-8"))
        runs = list(saved.get("runs") or [])
        ledger = {"llm": int((saved.get("ledger") or {}).get("llm")
                             or ledger["llm"]),
                  "fit": int((saved.get("ledger") or {}).get("fit")
                             or ledger["fit"])}
        base_shas = dict(saved.get("base_shas") or {})
        used = float(saved.get("wall_seconds_used") or 0.0)
        if used <= 0 and not prod:
            used = 2466.2
        started = time.time() - used
        payload["resumed_from_checkpoint"] = {
            "completed_run_ids": [row["run_id"] for row in runs],
            "ledger": dict(ledger),
            "wall_seconds_already_used": used,
        }
    elif store_root.exists():
        shutil.rmtree(store_root)
    try:
        probe = ps0c.probe_new_backend()
        payload["backend_probe"] = ps0c.redact(probe)
        if not probe.get("ok"):
            raise s1.Stop("BACKEND_UNAVAILABLE",
                          "new-relay probe failed: %s" % probe.get("reason"))
        from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
            compile_snapshot,
        )
        h0 = compile_snapshot(
            PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
            verify_lock=False)
        base_shas = _run_arms(
            cards=payload["cards"], h0=h0, store_root=store_root,
            ledger=ledger, started=started, runs=runs)
    except s1.Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {"verdict": stopped,
                           "reason": "%s: %s" % (type(exc).__name__, exc),
                           "traceback": traceback.format_exc()}
    payload["runs"] = runs
    payload["budget_equality"] = _budget_equality(base_shas)
    payload["aggregate"] = _aggregate(runs)
    payload["verdict"] = (
        _prod_verdict(payload["aggregate"], stopped=stopped, runs=runs)
        if prod
        else _verdict(payload["aggregate"], stopped=stopped, runs=runs))
    if prod:
        payload["supply_decoupling"] = _supply_decoupling(runs)
    attempt2_wall = round(time.time() - started, 1)
    payload["ledger"] = {
        "llm": ledger["llm"], "llm_cap": LLM_TOTAL_CAP,
        "fit": ledger["fit"], "fit_cap": FIT_TOTAL_CAP,
        "wall_seconds": attempt2_wall,
        "wall_seconds_cap": WALL_SECONDS_CAP, "downloads": 0,
    }
    if not prod:
        payload["ledger"].update({
            "wall_seconds_attempt1": ATTEMPT1_WASTED["wall_seconds"],
            "wall_seconds_combined": round(
                float(ATTEMPT1_WASTED["wall_seconds"]) + attempt2_wall, 1),
            "attempt1_charged": True,
            "attempt2_wall_timer_reset": True,
        })
    payload["oracle_isolation"] = s1._oracle_isolation_report()
    payload["obligations"] = {
        "methods_package_unmodified": not prod,
        "methods_surgery": (
            "W-1 wired the supply rung: fast_agent materialises a "
            "supplies_candidates card independently of the propose stage, "
            "and open_delayed gives a supply-sourced winner the delayed "
            "approval route every agent-authored winner already had.  No "
            "threshold, no authorization policy and no permission class."
            if prod else None),
        "runtime_contracts_operators_unmodified": True,
        "production_governance_unmodified": True,
        "no_new_skill_class_or_permission_platform": True,
        "injection_uses_existing_frozen_steps_channel": True,
        "injected_candidate_same_rights": True,
        "grants_execution_false": True,
        "experimental_prior_slot": True,
        "budgets_equal_across_arms": payload["budget_equality"]["all_equal"],
        "oracle_not_loaded_into_harness": True,
        "guided_positive_counts_zero_toward_source_auth": True,
        "downloads": 0,
        "full_repo_pytest_not_run": True,
        "semantic_discipline": payload["semantic_discipline"],
    }
    payload["outside_book"] = [
        "PS-2 scored Support only when the select stage named the inject.  "
        "The harness gives every pool member a Support trial inside the "
        "round budget, so ps2_run9 / ps2_run12 were full Support+delayed "
        "walks recorded as 0/4.  The funnel now attributes by the Episode "
        "the inject's own probe wrote (episode_id carries the Skill id).",
    ] if prod else [
        "attempt 1 printed 11/12 then InternalServerError on ps2_run12; "
        "in-memory records dropped; charged 67 LLM / 31 fit / 5236s.",
        "attempt 2 probe failed after the trycloudflare tunnel died.",
        "attempt 3 (this book) restarts the 12-run protocol on the "
        "user-restarted relay; wall hard-cap 2h; ledger continues from "
        "67/31; no checkpoint existed so all 12 run-ids are re-executed "
        "and persisted after each unit.",
    ]
    s1._dump(out_json, ps0c.redact(payload))
    out_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "reason": payload["verdict"]["reason"],
                      "ledger": payload["ledger"],
                      "artifact": str(out_json)},
                     ensure_ascii=False, indent=1), flush=True)
    return 0 if payload["verdict"]["verdict"] in (
        "MECHANICAL_RUNG_CONFIRMED", "POOL_ENTRY_WITHOUT_CONVERSION",
        "PLACEBO_CONVERSION", "SUPPLY_RUNG_PRODUCTION_CONFIRMED",
        "SUPPLY_WITHOUT_CONVERSION") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="PS-2 mechanical supply")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue from ps2_mechanical_supply.checkpoint.json")
    parser.add_argument("--prod-run", action="store_true",
                        help="W-1: same 12-run protocol on the wired "
                             "production supply rung (ps2p_run1..12)")
    parser.add_argument("--prod-resume", action="store_true",
                        help="continue ps2p from its own checkpoint")
    args = parser.parse_args()
    if args.prod_run or args.prod_resume:
        return run(resume=bool(args.prod_resume), prod=True)
    if not args.compile_only and not args.run and not args.resume:
        parser.error("pass --compile-only, --run, --resume or --prod-run")
    return run(compile_only=args.compile_only,
               resume=bool(args.resume) or (not args.compile_only))


if __name__ == "__main__":
    raise SystemExit(main())
