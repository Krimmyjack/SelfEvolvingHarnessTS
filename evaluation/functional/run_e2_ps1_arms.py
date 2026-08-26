"""PS-1 arms -- does a Scoped hypothesis change what the proposal stage names?

S1c named the bottleneck ``candidate discovery``.  On
``GunPointOldVersusYoung__impulse_v2`` the sealed oracle carries a +0.184
held-in headroom on ``hampel_filter`` and no arm ever proposed it.  This
runner asks whether a hypothesis compiled from two independent re-earned
positives, served as an ordinary ``SkillEntry`` through the existing
retrieval shape, raises the probability that the right Program family enters
the same small candidate budget -- and whether the Target's own feedback then
converts it or refuses it.

Architecture, per the arbitration: no new class, no fourth kind of Active
Skill, no permission platform.  The card is a ``SkillEntry`` and it uses the
authority fields ``methods/ttha/ordering_card.py`` already defines --
``reorders_supplied_candidates`` / ``supplies_candidates`` /
``suppresses_operators`` / ``grants_execution``.  PS-1 occupies the empty cell
``supplies_candidates=true, grants_execution=false``: a historical Skill may
put one candidate on the table, and may not execute it.

Three arms, identical budgets:

* ``A3``          -- no Source Skill at all;
* ``A5-neutral``  -- the same SkillEntry shape carrying no operator name and
                     no Program family, every authority flag false.  A pure
                     inert card: it measures the effect of a card *existing*;
* ``A5-scoped``   -- the same shape with the machine Scope matched and
                     ``supplies_candidates`` alone opened.

The prior slot is a runner-layer experiment mechanism only: the runner places
the card on the arm's own snapshot without a Slow authorization audit.
Everything downstream is production -- ``resolve_harness_view`` decides
visibility from ``observable_applicability``, the proposal stage still works
inside the one ``maximum_candidates`` cap, and neither card supplies a frozen
program, so no card can bypass the proposal stage or buy an extra search.

Evidence grade: development-mechanism, pilot.  ``GunPointOldVersusYoung``
shares ``GunPointFamily`` with source A, so this isolates a mechanism and is
not a cross-family transfer claim.  A guided positive counts zero toward any
cross-domain authorization for the Source Skill.
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
import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
PS0_JSON = E2 / "ps0_reearn_sources.json"
OUT_JSON = E2 / "ps1_proposal_shift_r2.json"
OUT_MD = E2 / "ps1_proposal_shift_r2.md"
CARD_DIR = E2 / "ps1_cards"

PROTOCOL_VERSION = "ps1_proposal_shift_v2_pilot"
EVIDENCE_GRADE = "development-mechanism (pilot)"

EXAM_UNIT = {"unit_id": "GunPointOldVersusYoung__impulse_v2",
             "dataset": "GunPointOldVersusYoung", "injection": "impulse_v2",
             "series_length": 150}
ARM_A3 = "A3"
ARM_NEUTRAL = "A5-neutral"
ARM_SCOPED = "A5-scoped"
ARMS = (ARM_A3, ARM_NEUTRAL, ARM_SCOPED)
REPLICATES = 4
LLM_PER_RUN = 12
FIT_PER_RUN = 10
ROUNDS = s1.HELD_IN_ROUNDS
TARGET_FAMILY = ps0.TARGET_FAMILY
TARGET_OPERATOR = ps0.TARGET_OPERATOR
MATERIAL = s1.MATERIAL

LLM_TOTAL_CAP = ps0.LLM_TOTAL_CAP
FIT_TOTAL_CAP = ps0.FIT_TOTAL_CAP
WALL_SECONDS_CAP = ps0.WALL_SECONDS_CAP

CARD_KIND = "ps1_source_hypothesis"
SCOPED_SKILL_ID = "ps1_source_hypothesis_scoped_v1"
NEUTRAL_SKILL_ID = "ps1_source_hypothesis_neutral_v1"
TOKEN_TOLERANCE = 0.15

# Interleaved so that any drift in the live model over the run sequence hits
# the three arms evenly.  Frozen before the first run.
RUN_PLAN = tuple(
    {"run_id": "ps1_run%d" % (index + 1),
     "arm": ARMS[index % len(ARMS)],
     "replicate": index // len(ARMS) + 1}
    for index in range(REPLICATES * len(ARMS))
)

AUTHORITY_FIELDS = ("reorders_supplied_candidates", "supplies_candidates",
                    "suppresses_operators", "grants_execution")


# =========================================================================== #
# Part 1 -- card compilation, mechanical from the Part 0 Scope
# =========================================================================== #
def _edit_schema_features() -> frozenset[str]:
    """Feature names the EditController schema will actually store.

    ``contracts/observables.OBSERVABLE_FEATURES`` is a superset of
    ``observable_feature_v1.json``.  A leaf that exists only in the Python
    table (level_region_*, outlier_region_end_fraction,
    level_only_post_shift_support_sufficient) is a legal Observation but
    not a legal edit-manifest predicate.  Dumping the raw intersection
    into SkillEntry.observable_applicability therefore fails shape
    validation before any arm runs.
    """
    schema_path = (PROJECT_ROOT / "contracts" / "schemas"
                   / "observable_feature_v1.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for option in schema.get("oneOf") or []:
        feature = (option.get("properties") or {}).get("feature") or {}
        if "const" in feature:
            names.add(str(feature["const"]))
        names.update(str(item) for item in (feature.get("enum") or []))
    return frozenset(names)


def _applicability_leaves(
        scope: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    legal = _edit_schema_features()
    leaves = [{"feature": "task_kind", "op": "==",
               "value": str(scope["task_kind"])}]
    dropped: list[str] = []
    for key, value in sorted(dict(scope["pattern_intersection"]).items()):
        if key not in legal:
            dropped.append(str(key))
            continue
        leaves.append({"feature": str(key), "op": "==", "value": value})
    return leaves, dropped


def _applicability(scope: Mapping[str, Any]) -> dict[str, Any]:
    leaves, _dropped = _applicability_leaves(scope)
    return {"all": leaves}


def _authority(*, supplies: bool) -> dict[str, Any]:
    return {
        "reorders_supplied_candidates": False,
        "supplies_candidates": bool(supplies),
        "suppresses_operators": False,
        "grants_execution": False,
    }


def _scoped_card(scope: Mapping[str, Any],
                 sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pattern = dict(scope["pattern_intersection"])
    geometry = list(scope["program_geometry"])
    provenance = "; ".join(
        "%s (Support %+.4f, delayed %+.4f)"
        % (row["unit_id"], float(row["support_gain"]),
           float(row["delayed_gain"]))
        for row in sources)
    body = "\n".join([
        "WHEN: task_kind == %s, consumer %s, metric %s, and the deployment-"
        "visible pattern reads %s."
        % (scope["task_kind"], scope["consumer_id"], scope["metric"],
           ", ".join("%s=%s" % (key, value)
                     for key, value in sorted(pattern.items()))),
        "OBSERVE: before proposing, read those same pattern features in the "
        "current Workspace and check whether the localized extreme deviation "
        "they describe is present here too.",
        "TRY-HYPOTHESIS: prioritise exploring the %s family (%s) among the "
        "candidates you propose. This is a ranking suggestion only and is not "
        "execution authority: it supplies one candidate for the same "
        "candidate budget and grants no right to deploy."
        % (TARGET_FAMILY, ", ".join(geometry)),
        "EVIDENCE: two independent prior domains improved under this family "
        "in the same direction -- %s. n = 2. Two agreeing domains is weak "
        "evidence; it establishes a hypothesis worth one probe, not a fact."
        % provenance,
        "VERIFY: the hypothesis holds here only if this Target's own held-in "
        "Support reads materially positive and the delayed feedback approves "
        "the Draft. Neither is assumed from the prior domains.",
        "FALLBACK: if Support or delayed refuses, drop the hypothesis and "
        "return to identity rather than retrying the family.",
    ])
    assert "Frozen program steps:" not in body
    return {
        "schema_version": "skill-entry/1",
        "skill_id": SCOPED_SKILL_ID,
        "skill_kind": "capability",
        "revision": 1,
        "body": body,
        "observable_applicability": _applicability(scope),
        "allowed_tools": [],
        "risk_guards": {
            "card_kind": CARD_KIND,
            "authority": _authority(supplies=True),
            "scope_v1": _plainify(scope),
            "evidence": {
                "source_count": len(sources),
                "sources": [
                    {"unit_id": row["unit_id"], "run_id": row.get("run_id"),
                     "support_gain": row["support_gain"],
                     "delayed_gain": row["delayed_gain"],
                     "direction": "improved"} for row in sources],
                "uncertainty": ("n=2 agreeing domains; a hypothesis worth one "
                                "probe, not a fact"),
            },
            "requires_target_support": True,
            "execution_right": "withheld_supplies_candidate_only",
            "counting_rule": (
                "a positive earned under this card is a Target-local Skill "
                "only and counts zero toward any cross-domain authorization "
                "for this Source Skill"),
        },
    }


def _neutral_card(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Same shape, same Scope, no direction of any kind, all flags false."""
    pattern = dict(scope["pattern_intersection"])
    body = "\n".join([
        "WHEN: task_kind == %s, consumer %s, metric %s, and the deployment-"
        "visible pattern reads %s."
        % (scope["task_kind"], scope["consumer_id"], scope["metric"],
           ", ".join("%s=%s" % (key, value)
                     for key, value in sorted(pattern.items()))),
        "OBSERVE: before proposing, read the deployment-visible features of "
        "the current Workspace and note what the readings are.",
        "TRY-HYPOTHESIS: this card records no candidate and expresses no "
        "preference among the available options. It is a ranking suggestion "
        "about nothing and is not execution authority: it supplies no "
        "candidate to the candidate budget and grants no right to deploy.",
        "EVIDENCE: this card carries no accumulated outcome evidence from any "
        "prior domain and reports no direction. n = 0. No evidence is weak "
        "evidence; it establishes nothing worth a probe, and no fact.",
        "VERIFY: whatever is proposed here holds only if this Target's own "
        "held-in Support reads materially positive and the delayed feedback "
        "approves the Draft. Nothing is assumed from anywhere else.",
        "FALLBACK: if Support or delayed refuses, drop the candidate and "
        "return to identity rather than retrying the same option.",
    ])
    assert "Frozen program steps:" not in body
    return {
        "schema_version": "skill-entry/1",
        "skill_id": NEUTRAL_SKILL_ID,
        "skill_kind": "capability",
        "revision": 1,
        "body": body,
        "observable_applicability": _applicability(scope),
        "allowed_tools": [],
        "risk_guards": {
            "card_kind": CARD_KIND,
            "authority": _authority(supplies=False),
            "scope_v1": _plainify(scope),
            "evidence": {"source_count": 0, "sources": [],
                         "uncertainty": "no accumulated evidence"},
            "requires_target_support": True,
            "execution_right": "withheld_inert_control_card",
            "counting_rule": (
                "control card; it authorizes nothing and counts toward "
                "nothing"),
        },
    }


def _plainify(value: Any) -> Any:
    return s1._plain(value)


def _tokens(text: str) -> int:
    return len(str(text).split())


def _card_audit(scoped: Mapping[str, Any],
                neutral: Mapping[str, Any]) -> dict[str, Any]:
    """Everything that must be true of the pair before an arm runs."""
    from SelfEvolvingHarnessTS.operators.registry import OPERATOR_NAMES

    neutral_blob = json.dumps(neutral, ensure_ascii=False).lower()
    named_ops = sorted(op for op in OPERATOR_NAMES if op in neutral_blob)
    family_words = sorted(
        family for family in s1.PROGRAM_FAMILIES if family in neutral_blob)
    scoped_tokens = _tokens(scoped["body"])
    neutral_tokens = _tokens(neutral["body"])
    ratio = neutral_tokens / float(scoped_tokens or 1)
    scope = scoped["risk_guards"]["scope_v1"]
    machine_leaves, dropped = _applicability_leaves(scope)
    return {
        "scoped_body_tokens": scoped_tokens,
        "neutral_body_tokens": neutral_tokens,
        "token_ratio": round(ratio, 4),
        "token_ratio_within_tolerance": abs(ratio - 1.0) <= TOKEN_TOLERANCE,
        "neutral_names_no_operator": not named_ops,
        "neutral_operator_hits": named_ops,
        "neutral_names_no_program_family": not family_words,
        "neutral_family_hits": family_words,
        "neutral_all_authority_false": all(
            neutral["risk_guards"]["authority"][field] is False
            for field in AUTHORITY_FIELDS),
        "scoped_authority": dict(scoped["risk_guards"]["authority"]),
        "neutral_authority": dict(neutral["risk_guards"]["authority"]),
        "scoped_opens_only_supplies_candidates": (
            scoped["risk_guards"]["authority"]["supplies_candidates"] is True
            and all(scoped["risk_guards"]["authority"][field] is False
                    for field in AUTHORITY_FIELDS
                    if field != "supplies_candidates")),
        "neither_card_supplies_a_frozen_program": all(
            not card["allowed_tools"]
            and "Frozen program steps:" not in card["body"]
            for card in (scoped, neutral)),
        "identical_applicability": (
            scoped["observable_applicability"]
            == neutral["observable_applicability"]),
        "same_schema_and_kind": (
            scoped["schema_version"] == neutral["schema_version"]
            and scoped["skill_kind"] == neutral["skill_kind"]),
        "machine_applicability_leaf_count": len(machine_leaves),
        "pattern_leaves_dropped_as_uncontracted_for_edit_schema": dropped,
        "dropped_leaves_remain_in_body_and_scope_v1": True,
    }


def _budget_equality(base_shas: Mapping[str, str]) -> dict[str, Any]:
    """Evidence that the prior slot buys no extra search."""
    constraints = cls._task_context().deployment_constraints
    per_arm = {
        arm: {
            "maximum_candidates": int(constraints.maximum_candidates),
            "maximum_modified_fraction": float(
                constraints.maximum_modified_fraction),
            "support_trial_budget_per_round": int(cls.SUPPORT_TRIAL_BUDGET),
            "rounds": list(ROUNDS),
            "llm_cap_per_run": LLM_PER_RUN,
            "fit_cap_per_run": FIT_PER_RUN,
            "base_runtime_bundle_sha": base_shas.get(arm),
        } for arm in ARMS
    }
    keys = ("maximum_candidates", "maximum_modified_fraction",
            "support_trial_budget_per_round", "rounds", "llm_cap_per_run",
            "fit_cap_per_run")
    equal = {key: len({json.dumps(per_arm[arm][key], sort_keys=True)
                       for arm in ARMS}) == 1 for key in keys}
    return {
        "per_arm": per_arm,
        "equal_across_arms": equal,
        "all_equal": all(equal.values()),
        "why_a_supplied_candidate_costs_a_slot": (
            "the card carries no frozen program and no allowed tool, so it "
            "cannot reach _skill_frozen_candidates or _frozen_recall.  Any "
            "candidate it inspires is proposed by the same proposal stage and "
            "counts inside the same maximum_candidates cap; nothing is added "
            "outside it."),
        "base_snapshots_differ_only_by_the_card": {
            arm: base_shas.get(arm) for arm in ARMS},
    }


# =========================================================================== #
# Part 2 -- the arms
# =========================================================================== #
def _funnel(result: Mapping[str, Any]) -> dict[str, Any]:
    proposed = selected = verified = supported = approved = deployed = False
    detail: list[dict[str, Any]] = []
    for record in result.get("rounds") or []:
        for row in record.get("proposals") or []:
            if row["family"] != TARGET_FAMILY:
                continue
            proposed = True
            selected = selected or bool(row["chosen_by_select"])
            if row["outcome"] == "probe":
                verified = True
            detail.append({"round": record.get("round"), **row})
        for episode in record.get("episodes") or []:
            if str(episode.get("workflow_signature")) != TARGET_OPERATOR:
                continue
            support = episode.get("support_gain")
            if support is not None and float(support) >= MATERIAL \
                    and str(episode.get("relation")) == "POSITIVE":
                supported = True
        if record.get("winner_delayed_approved"):
            winner_ops = [str(step.get("op"))
                          for step in (record.get("winner_program") or [])]
            if TARGET_OPERATOR in winner_ops:
                approved = True
    deployed_ops = [str(step.get("op")) for step
                    in (result.get("deployment") or {}).get(
                        "applied_program") or []]
    deployed = TARGET_OPERATOR in deployed_ops
    return {
        "proposed": proposed,
        "selected_by_agent": selected,
        "passed_verifier": verified,
        "support_material_positive": supported,
        "delayed_approved": approved,
        "deployed": deployed,
        "target_proposals": detail,
    }


def _cost_to_first_effective_skill(result: Mapping[str, Any]) -> dict[str, Any]:
    """Rounds, probes, fits and LLM spent before a Skill was actually earned."""
    rounds = 0
    probes = 0
    for record in result.get("rounds") or []:
        rounds += 1
        probes += len(record.get("probes") or [])
        if record.get("winner_delayed_approved"):
            return {"reached": True, "rounds": rounds, "probes": probes,
                    "consumer_fits": record.get("consumer_fits_after"),
                    "llm_calls_cumulative": None}
    return {"reached": False, "rounds": rounds, "probes": probes,
            "consumer_fits": result.get("consumer_fits"),
            "llm_calls_cumulative": result.get("llm_calls")}


def _run_arms(*, cards: Mapping[str, Any], h0: Any, store_root: Path,
              ledger: dict[str, int], started: float) -> list[dict[str, Any]]:
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
    runs: list[dict[str, Any]] = []
    for plan in RUN_PLAN:
        if ledger["llm"] >= LLM_TOTAL_CAP or ledger["fit"] >= FIT_TOTAL_CAP:
            raise s1.Stop("COMPUTE_BUDGET_EXCEEDED",
                          "book cap reached before %s" % plan["run_id"])
        if time.time() - started > WALL_SECONDS_CAP:
            raise s1.Stop("COMPUTE_BUDGET_EXCEEDED",
                          "wall clock cap reached before %s" % plan["run_id"])
        arm = plan["arm"]
        backend = cls._live_backend(LLM_PER_RUN)
        result = s1.run_unit(
            unit=EXAM_UNIT, cell=cell, arm=arm, base_snapshot=bases[arm],
            carried_episodes=(), agent_factory=cls._live_agent,
            backend=backend, store_root=store_root / plan["run_id"],
            rounds=ROUNDS, fit_cap=FIT_PER_RUN)
        ledger["llm"] += int(result.get("llm_calls") or 0)
        ledger["fit"] += int(result.get("consumer_fits") or 0)
        public = s1._public_unit_result(result)
        funnel = _funnel(public)
        card_seen = sorted({
            skill_id for record in public.get("rounds") or []
            for skill_id in record.get("retrieved_skill_ids") or []
            if skill_id in (SCOPED_SKILL_ID, NEUTRAL_SKILL_ID)})
        deployment = public.get("deployment") or {}
        deltas = deployment.get("heldout_recall_delta_by_class") or {}
        runs.append({
            "run_id": plan["run_id"],
            "arm": arm,
            "replicate": plan["replicate"],
            "base_runtime_bundle_sha": base_shas[arm],
            "card_in_fast_view": card_seen,
            "funnel": funnel,
            "proposal_ledger": ps0._proposal_ledger(public),
            "proposal_families": sorted({
                row["family"] for record in public.get("rounds") or []
                for row in record.get("proposals") or []
                if row["family"] != "identity"}),
            "distinct_non_target_families": sorted({
                row["family"] for record in public.get("rounds") or []
                for row in record.get("proposals") or []
                if row["family"] not in ("identity", TARGET_FAMILY)}),
            "cost_to_first_effective_skill": _cost_to_first_effective_skill(
                public),
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
                "heldout_accuracy_gain": deployment.get(
                    "heldout_accuracy_gain"),
                "worst_class_delta": (min(float(value)
                                          for value in deltas.values())
                                      if deltas else 0.0),
            },
            "rounds": public.get("rounds"),
        })
        print("%-11s %-11s proposed=%-5s support=%-5s approved=%-5s "
              "deployed=%-5s gain=%+.4f llm=%s fits=%s"
              % (plan["run_id"], arm, funnel["proposed"],
                 funnel["support_material_positive"],
                 funnel["delayed_approved"], funnel["deployed"],
                 float(deployment.get("heldout_accuracy_gain") or 0.0),
                 result.get("llm_calls"), result.get("consumer_fits")),
              flush=True)
    return runs, base_shas


# =========================================================================== #
# Part 3 -- aggregate and verdict
# =========================================================================== #
def _aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in runs if row["arm"] == arm]
        if not rows:
            continue
        stages = ("proposed", "selected_by_agent", "passed_verifier",
                  "support_material_positive", "delayed_approved", "deployed")
        out[arm] = {
            "runs": len(rows),
            "run_ids": [row["run_id"] for row in rows],
            "funnel_counts": {stage: sum(1 for row in rows
                                         if row["funnel"][stage])
                              for stage in stages},
            "target_family_proposal_rate": "%d/%d" % (
                sum(1 for row in rows if row["funnel"]["proposed"]), len(rows)),
            "card_served_runs": sum(1 for row in rows
                                    if row["card_in_fast_view"]),
            "mean_llm": round(sum(int(row["llm_calls"] or 0)
                                  for row in rows) / len(rows), 2),
            "mean_fits": round(sum(int(row["consumer_fits"] or 0)
                                   for row in rows) / len(rows), 2),
            "mean_probes": round(sum(int(row["probes"] or 0)
                                     for row in rows) / len(rows), 2),
            "wasted_probes": sum(int(row["wasted_probes"] or 0)
                                 for row in rows),
            "deployed_utilities": [row["deployment"]["heldout_accuracy_gain"]
                                   for row in rows],
            "worst_class_deltas": [row["deployment"]["worst_class_delta"]
                                   for row in rows],
            "harm_runs": sum(1 for row in rows
                             if float(row["deployment"]["worst_class_delta"])
                             <= -MATERIAL),
            "distinct_non_target_families": sorted({
                family for row in rows
                for family in row["distinct_non_target_families"]}),
            "runs_reaching_an_effective_skill": sum(
                1 for row in rows
                if row["cost_to_first_effective_skill"]["reached"]),
            "cost_to_first_effective_skill": [
                {"run_id": row["run_id"],
                 **row["cost_to_first_effective_skill"]}
                for row in rows if row["cost_to_first_effective_skill"][
                    "reached"]],
        }
    return out


def _verdict(aggregate: Mapping[str, Any], *,
             stopped: str | None) -> dict[str, Any]:
    if stopped:
        return {"verdict": stopped, "reason": "stopped before the full plan"}
    if not all(arm in aggregate for arm in ARMS):
        return {"verdict": "COMPUTE_BUDGET_EXCEEDED",
                "reason": "not every arm completed its replicates"}

    def rate(arm: str) -> int:
        return int(aggregate[arm]["funnel_counts"]["proposed"])

    scoped, neutral, a3 = rate(ARM_SCOPED), rate(ARM_NEUTRAL), rate(ARM_A3)
    n = aggregate[ARM_SCOPED]["runs"]
    placebo = abs(neutral - a3) >= 2
    separated = scoped >= 3 and max(neutral, a3) <= 1
    gray = (not separated) and scoped > max(neutral, a3)
    converted = int(aggregate[ARM_SCOPED]["funnel_counts"]["deployed"]) >= 1 \
        and int(aggregate[ARM_SCOPED]["funnel_counts"][
            "delayed_approved"]) >= 1
    harm_not_up = (aggregate[ARM_SCOPED]["harm_runs"]
                   <= max(aggregate[ARM_NEUTRAL]["harm_runs"],
                          aggregate[ARM_A3]["harm_runs"]))
    facts = {
        "proposal_rate": {arm: "%d/%d" % (rate(arm), aggregate[arm]["runs"])
                          for arm in ARMS},
        "separated": separated,
        "gray_zone": gray,
        "converted_to_deployment": converted,
        "harm_not_increased": harm_not_up,
        "placebo_separation": placebo,
    }
    if placebo:
        return {"verdict": "PLACEBO_EFFECT", "facts": facts,
                "reason": ("A5-neutral departs from A3 by %d of %d runs on "
                           "proposal rate, so the presence of a card changes "
                           "behaviour on its own and any scoped effect has to "
                           "be read against that baseline, not against A3"
                           % (abs(neutral - a3), n))}
    if separated and converted and harm_not_up:
        return {"verdict": "PROPOSAL_SHIFT_CONFIRMED", "facts": facts,
                "reason": ("A5-scoped proposed the target family in %d of %d "
                           "runs against at most %d in either control, at "
                           "least one run walked the whole funnel to "
                           "deployment, and harm did not rise"
                           % (scoped, n, max(neutral, a3)))}
    if separated and not converted:
        return {"verdict": "SHIFT_WITHOUT_CONVERSION", "facts": facts,
                "reason": ("proposal rate separated (%d/%d against at most "
                           "%d) but nothing converted; the funnel counts name "
                           "the layer it stopped at"
                           % (scoped, n, max(neutral, a3)))}
    if gray:
        return {"verdict": "SHIFT_WEAK", "facts": facts,
                "reason": ("A5-scoped leads the controls (%d/%d against %d "
                           "and %d) but does not reach the pre-registered "
                           "separation.  Per the arbitration this pilot does "
                           "not append a batch; a fuller replication is a "
                           "separate pre-frozen experiment"
                           % (scoped, n, neutral, a3))}
    return {"verdict": "NO_PROPOSAL_SHIFT", "facts": facts,
            "reason": ("the three arms' proposal distributions overlap: %s"
                       % facts["proposal_rate"])}


def _markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = [
        "# PS-1 -- proposal shift under a Scoped hypothesis (pilot)",
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
        "> Pilot grade.  This result freezes no production design.  "
        "%s shares GunPointFamily with source A, so it isolates a mechanism "
        "and is not a cross-family transfer claim.  A positive earned under "
        "the card is a Target-local Skill and counts zero toward any "
        "cross-domain authorization for the Source Skill."
        % EXAM_UNIT["unit_id"],
        "",
        "## Cards (SkillEntry, existing authority fields)",
        "",
        "| field | A5-scoped | A5-neutral |",
        "|---|---|---|",
    ]
    scoped = payload["cards"]["scoped"]
    neutral = payload["cards"]["neutral"]
    for field in ("skill_id", "schema_version", "skill_kind", "revision"):
        lines.append("| %s | `%s` | `%s` |" % (field, scoped[field],
                                               neutral[field]))
    lines.append("| allowed_tools | %s | %s |" % (scoped["allowed_tools"],
                                                  neutral["allowed_tools"]))
    for field in AUTHORITY_FIELDS:
        lines.append("| authority.%s | **%s** | **%s** |" % (
            field, scoped["risk_guards"]["authority"][field],
            neutral["risk_guards"]["authority"][field]))
    lines.append("| observable_applicability | %s | %s |" % (
        json.dumps(scoped["observable_applicability"], ensure_ascii=False),
        "identical" if scoped["observable_applicability"]
        == neutral["observable_applicability"] else json.dumps(
            neutral["observable_applicability"], ensure_ascii=False)))
    audit = payload["card_audit"]
    lines += ["", "### Card audit", ""]
    for key, value in audit.items():
        lines.append("- **%s**: %s" % (key, value))
    budget = payload["budget_equality"]
    lines += ["", "## Budget equality across the three arms", "",
              "- all equal: **%s**" % budget["all_equal"], ""]
    for key, value in budget["equal_across_arms"].items():
        lines.append("- %s equal: %s (value %s)"
                     % (key, value, budget["per_arm"][ARM_A3][key]))
    lines += ["", "- %s" % budget["why_a_supplied_candidate_costs_a_slot"], ""]
    lines += ["## Per-run readout", "",
              "| run | arm | card served | proposed | selected | verifier | "
              "Support | delayed | deployed | gain | worst-class | LLM | fits "
              "| probes |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in payload["runs"]:
        funnel = row["funnel"]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %+.4f | %+.4f | "
            "%s | %s | %s |" % (
                row["run_id"], row["arm"],
                "yes" if row["card_in_fast_view"] else "-",
                funnel["proposed"], funnel["selected_by_agent"],
                funnel["passed_verifier"],
                funnel["support_material_positive"],
                funnel["delayed_approved"], funnel["deployed"],
                float(row["deployment"]["heldout_accuracy_gain"] or 0.0),
                float(row["deployment"]["worst_class_delta"] or 0.0),
                row["llm_calls"], row["consumer_fits"], row["probes"]))
    lines += ["", "## Three-arm aggregate", "",
              "| arm | proposal rate | selected | verifier | Support | "
              "delayed | deployed | harm runs | mean LLM | mean fits | mean "
              "probes | other families |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        row = payload["aggregate"].get(arm)
        if not row:
            continue
        counts = row["funnel_counts"]
        lines.append("| %s | **%s** | %d | %d | %d | %d | %d | %d | %s | %s | "
                     "%s | %s |" % (
                         arm, row["target_family_proposal_rate"],
                         counts["selected_by_agent"], counts["passed_verifier"],
                         counts["support_material_positive"],
                         counts["delayed_approved"], counts["deployed"],
                         row["harm_runs"], row["mean_llm"], row["mean_fits"],
                         row["mean_probes"],
                         ", ".join(row["distinct_non_target_families"])
                         or "none"))
    lines += ["", "### Cost to the first effective Skill", ""]
    for arm in ARMS:
        row = payload["aggregate"].get(arm)
        if not row:
            continue
        reached = row["cost_to_first_effective_skill"]
        lines.append("- **%s**: %d of %d runs reached one%s"
                     % (arm, row["runs_reaching_an_effective_skill"],
                        row["runs"],
                        ("; " + "; ".join(
                            "`%s` after %d round(s), %d probe(s), %s fit(s)"
                            % (item["run_id"], item["rounds"], item["probes"],
                               item["consumer_fits"]) for item in reached))
                        if reached else ""))
    ledger = payload["ledger"]
    lines += ["", "## Cost", "",
              "- LLM: %d / %d" % (ledger["llm"], ledger["llm_cap"]),
              "- Consumer fits: %d / %d" % (ledger["fit"], ledger["fit_cap"]),
              "- wall clock: %.1f s / %d s"
              % (ledger["wall_seconds"], ledger["wall_seconds_cap"]),
              "- downloads: 0", "", "## Obligations", ""]
    for key, value in payload["obligations"].items():
        lines.append("- **%s**: %s" % (key, value))
    if payload.get("outside_book"):
        lines += ["", "## Outside the book", ""]
        lines += ["- %s" % item for item in payload["outside_book"]]
    return "\n".join(lines) + "\n"


def _halt_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    analysis = payload.get("miss_analysis") or {}
    lines = [
        "# PS-1 halted before Part 3",
        "",
        "protocol: `%s`  git: `%s`" % (payload["protocol_version"],
                                       payload["git_head"]),
        "",
        "**%s**" % verdict["verdict"],
        "",
        verdict["reason"],
        "",
        "No card was compiled and no arm ran.  Part 3 is conditional on both "
        "scenes re-earning; a card compiled from one source would be a "
        "single-domain claim, which is exactly what the two-source rule "
        "exists to prevent.",
        "",
        "## Where each re-earn attempt broke",
        "",
        "| scene | run | earned | proposed | selected | verifier | Support | "
        "delayed | broke at | families proposed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in analysis.get("per_run", []):
        funnel = row["funnel"]
        lines.append("| %s | `%s` | %s | %s | %s | %s | %s | %s | **%s** | %s |"
                     % (row["scene"], row["run_id"], row["earned"],
                        funnel["proposed"], funnel["selected_by_agent"],
                        funnel["passed_verifier"],
                        funnel["support_material_positive"],
                        funnel["delayed_approved"], row["broke_at"] or "-",
                        ", ".join(row["families_proposed"]) or "none"))
    lines += ["", "### The target family's own readings", "",
              "| run | round | relation | Support | delayed |",
              "|---|---|---|---|---|"]
    for row in analysis.get("per_run", []):
        for read in row["target_family_reads"]:
            lines.append("| `%s` | %s | %s | %s | %s |" % (
                row["run_id"], read["round"], read["relation"],
                read["support_gain"], read["delayed_gain"]))
    lines += ["", "- never proposed: %s"
              % (analysis.get("runs_where_the_family_was_never_proposed")
                 or "none"),
              "- proposed but Support refused: %s"
              % (analysis.get("runs_where_it_was_proposed_but_support_refused")
                 or "none"),
              "- %s" % analysis.get("selection_is_not_on_the_critical_path",
                                    ""),
              "", "**Reading**: %s" % analysis.get("reading", ""), "",
              "## Obligations", ""]
    for key, value in payload["obligations"].items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


def miss_analysis(ps0_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Where each re-earn attempt actually broke, off the repaired ledger.

    This is the reading the PS-0 record repair exists to make possible: S1c
    could only say "the family was never proposed", because it tagged
    proposals by scanning the agent's invented candidate id for an operator
    word and always found none.
    """
    stages = ("proposed", "selected_by_agent", "passed_verifier",
              "support_material_positive", "delayed_approved")
    # ``selected_by_agent`` is reported but is not on the critical path: a
    # candidate the agent did not choose is still probed if the Support budget
    # reaches it, and source A' earned exactly that way.
    critical = ("proposed", "passed_verifier", "support_material_positive",
                "delayed_approved")
    rows: list[dict[str, Any]] = []
    for scene in ps0_payload.get("scenes") or []:
        for run in scene.get("runs") or []:
            funnel = _funnel(run)
            earned = bool(run["earned"]["earned"])
            broke_at = None if earned else next(
                (stage for stage in critical if not funnel[stage]), None)
            target_reads = [
                {"round": record.get("round"),
                 "relation": episode.get("relation"),
                 "support_gain": episode.get("support_gain"),
                 "delayed_gain": episode.get("delayed_gain")}
                for record in run.get("rounds") or []
                for episode in record.get("episodes") or []
                if str(episode.get("workflow_signature")) == TARGET_OPERATOR]
            rows.append({
                "scene": scene["scene"],
                "run_id": run["run_id"],
                "earned": earned,
                "funnel": {stage: funnel[stage] for stage in stages},
                "broke_at": broke_at,
                "families_proposed": run["proposal_families"],
                "target_family_reads": target_reads,
                "rounds_in_which_target_was_proposed": sorted({
                    row["round"] for row in run["proposal_ledger"]
                    if row["family"] == TARGET_FAMILY}),
            })
    proposed_but_unconfirmed = [row for row in rows
                                if row["funnel"]["proposed"]
                                and not row["funnel"][
                                    "support_material_positive"]]
    never_proposed = [row for row in rows if not row["funnel"]["proposed"]]
    return {
        "per_run": rows,
        "runs_where_the_family_was_never_proposed": [
            row["run_id"] for row in never_proposed],
        "runs_where_it_was_proposed_but_support_refused": [
            row["run_id"] for row in proposed_but_unconfirmed],
        "selection_is_not_on_the_critical_path": (
            "source A' earned without the agent ever choosing the family: it "
            "chose a level-shift candidate, that candidate was refused by the "
            "verifier, and the Support budget reached the second entry in "
            "probe_order, which was the hampel one.  Selection is reported "
            "but a run is not counted as broken there."),
        "reading": (
            "the two misses do not share a bottleneck.  One never named the "
            "family at all, which is the discovery failure S1c described.  "
            "The other named it, probed it, and the Target's own Support read "
            "exactly 0.0 -- a confirmation failure, not a discovery failure.  "
            "A hypothesis card can only address the first kind."),
    }


def run() -> int:
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)
    if not PS0_JSON.is_file():
        raise SystemExit("run PS-0 first: %s missing" % PS0_JSON)
    ps0_payload = json.loads(PS0_JSON.read_text(encoding="utf-8"))
    part0 = ps0_payload["part0_reverify"]
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "exam_unit": EXAM_UNIT["unit_id"],
        "ps0_source": PS0_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "part0_reverify": part0,
        "run_plan": [dict(plan) for plan in RUN_PLAN],
        "arms": {
            ARM_A3: "no Source Skill",
            ARM_NEUTRAL: ("same SkillEntry shape, no operator name, no "
                          "Program family, every authority flag false"),
            ARM_SCOPED: ("same shape, Scope matched, supplies_candidates "
                         "true and grants_execution false"),
        },
        "experimental_prior_slot": True,
        "prior_slot_implementation": (
            "the runner places the SkillEntry on the arm's own snapshot via "
            "the frozen EditController path, without a Slow authorization "
            "audit.  Everything after that is production: resolve_harness_view "
            "decides visibility from observable_applicability and agent_core "
            "renders the card through the ordinary _skill_prompt.  No file "
            "under methods/, runtime/, contracts/ or operators/ is modified."),
    }
    if not part0.get("pass"):
        payload["verdict"] = {
            "verdict": part0.get("verdict", "PS1_SOURCES_NOT_REEARNED"),
            "reason": part0.get("reason", "Part 0 did not pass")}
        payload["miss_analysis"] = miss_analysis(ps0_payload)
        payload["ledger"] = {"llm": 0, "llm_cap": LLM_TOTAL_CAP, "fit": 0,
                             "fit_cap": FIT_TOTAL_CAP, "wall_seconds": 0.0,
                             "wall_seconds_cap": WALL_SECONDS_CAP}
        payload["obligations"] = {
            "arms_run": 0,
            "cards_compiled": False,
            "why_no_card": ("Part 3 is conditional on both scenes re-earning; "
                            "compiling a card from one source would make it a "
                            "single-domain claim, which is the thing the "
                            "two-source rule exists to prevent"),
            "llm_calls": 0,
            "methods_package_unmodified": True,
            "production_governance_unmodified": True,
            "stage_report_not_written": True,
        }
        s1._dump(OUT_JSON, payload)
        OUT_MD.write_text(_halt_markdown(payload), encoding="utf-8")
        print(json.dumps({"verdict": payload["verdict"]["verdict"],
                          "reason": payload["verdict"]["reason"],
                          "artifact": str(OUT_JSON)},
                         ensure_ascii=False, indent=1))
        return 1

    scope = part0["scope_v1"]
    sources = [
        {"unit_id": row["unit_id"], "run_id": row["run_id"],
         "support_gain": next(
             scene["earned"]["support_gain"] for scene in ps0_payload["scenes"]
             if scene["scene"] == row["scene"]),
         "delayed_gain": next(
             scene["earned"]["delayed_gain"] for scene in ps0_payload["scenes"]
             if scene["scene"] == row["scene"])}
        for row in part0["per_source"]]
    scoped = _scoped_card(scope, sources)
    neutral = _neutral_card(scope)
    payload["cards"] = {"scoped": scoped, "neutral": neutral}
    payload["card_audit"] = _card_audit(scoped, neutral)
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    for name, card in (("scoped", scoped), ("neutral", neutral)):
        (CARD_DIR / ("ps1_card_%s.json" % name)).write_text(
            json.dumps(s1._plain(card), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")

    ledger = {"llm": 0, "fit": 0}
    tag = "ps1_arms"
    store_root = Path(tempfile.gettempdir()) / tag
    if store_root.exists():
        shutil.rmtree(store_root)
    stopped: str | None = None
    runs: list[dict[str, Any]] = []
    base_shas: dict[str, str] = {}
    try:
        payload["backend_probe"] = s1._probe_live_backend()
        if not payload["backend_probe"].get("ok"):
            raise s1.Stop("INSTRUMENT_UNREADABLE", "backend probe failed")
        k0 = s1.compile_k0(store_root / "k0")
        runs, base_shas = _run_arms(cards=payload["cards"], h0=k0["h0"],
                                    store_root=store_root, ledger=ledger,
                                    started=started)
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
    payload["verdict"] = _verdict(payload["aggregate"], stopped=stopped)
    payload["ledger"] = {
        "llm": ledger["llm"], "llm_cap": LLM_TOTAL_CAP,
        "fit": ledger["fit"], "fit_cap": FIT_TOTAL_CAP,
        "wall_seconds": round(time.time() - started, 1),
        "wall_seconds_cap": WALL_SECONDS_CAP, "downloads": 0,
    }
    payload["oracle_isolation"] = s1._oracle_isolation_report()
    payload["obligations"] = {
        "methods_package_unmodified": True,
        "runtime_contracts_operators_unmodified": True,
        "production_governance_unmodified": True,
        "no_new_skill_class_or_permission_platform": True,
        "card_is_a_plain_skill_entry": True,
        "experimental_prior_slot": True,
        "budgets_equal_across_arms": payload["budget_equality"]["all_equal"],
        "neither_card_supplies_a_frozen_program": payload["card_audit"][
            "neither_card_supplies_a_frozen_program"],
        "guided_positive_counts_zero_toward_cross_domain_authorization": True,
        "pilot_grade_freezes_no_production_design": True,
        "gray_zone_appends_no_batch": True,
        "arms_run": len(runs),
        "downloads": 0,
        "oracle_isolation_holds": payload["oracle_isolation"]["holds"],
        "stage_report_not_written": True,
        "full_repo_pytest_not_run": True,
    }
    payload["outside_book"] = []
    s1._dump(OUT_JSON, payload)
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "facts": payload["verdict"].get("facts"),
                      "llm": ledger["llm"], "fits": ledger["fit"],
                      "seconds": payload["ledger"]["wall_seconds"],
                      "artifact": str(OUT_JSON)},
                     ensure_ascii=False, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.run:
        return run()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
