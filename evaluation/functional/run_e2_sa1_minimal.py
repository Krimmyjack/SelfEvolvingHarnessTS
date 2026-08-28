"""SA-1 minimal: Skill as an updatable hypothesis, wired and examined once.

Four stages, each gated, in order:

* **Part 0**  four attribution fields (implemented in the shared runner and in
  ``online_loop``); the gate here backfills them onto the recorded L1 replay.
* **Part 0.5**  Scope rule v2 -- a supply card's Pattern axis becomes the
  defect *family* S1a already froze, not one Episode's whole recorded view.
* **Part 1**  the three write-backs (R1 evidence, R2 narrowing, R3 demotion),
  replayed offline against L1's own ledger and two historical negatives.
* **Part 2**  one live six-unit course, three arms, revision on in exactly one.

  python evaluation/functional/run_e2_sa1_minimal.py --gates
  python evaluation/functional/run_e2_sa1_minimal.py --freeze
  python evaluation/functional/run_e2_sa1_minimal.py --run --seed r1
  python evaluation/functional/run_e2_sa1_minimal.py --resume --seed r1
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

import numpy as np  # noqa: E402

import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_s1v2_forward_course as sv  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    evaluate_applicability,
)
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    skill_revision as sr,
    source_skill as ss,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
GATES_JSON = E2 / "sa1_minimal_gates.json"
FREEZE_JSON = E2 / "sa1_course_freeze.json"
OUT_JSON = E2 / "sa1_minimal_r1.json"
OUT_MD = E2 / "sa1_minimal_r1.md"
CHECKPOINT = E2 / "sa1_minimal_r1.checkpoint.json"

S1A_AUDIT = E2 / "s1a_curriculum_audit.json"
L1_R1 = E2 / "l1_ladder_v2_replay_r1.json"
V3_RUN = E2 / "s1v2_v3_forward_run1.json"
ECG200_HARM = E2 / "t6_cls_conf_dev_ecg200.json"

PROTOCOL_VERSION = "sa1_minimal_v1"
EVIDENCE_GRADE = "development"
SA1_SKILL_ID = "sa1_supply_scope_v2"
TARGET_PROGRAM = "hampel_filter"

# Arm identities are the frozen strings the shared runner branches on; the
# report labels them as the book names them.
ARM_A3 = s1.ARM_A3
ARM_K0 = s1.ARM_K0
ARM_A5 = s1.ARM_A5
ARM_LABEL = {ARM_A3: "A3-reset", ARM_K0: "K0-fixed", ARM_A5: "A5-adaptive"}
ARMS = (ARM_A3, ARM_K0, ARM_A5)

# The re-encounter slot is a mechanism probe, not a sixth distinct domain.
REENCOUNTER_POSITION = 5

LLM_CAP = 150
FIT_CAP = 300
FIT_PER_UNIT_PER_ARM = sv.FIT_PER_UNIT_PER_ARM
WALL_SECONDS_CAP = int(4.2 * 60 * 60)
HALF_ROUNDS = sv.HALF_ROUNDS
DELTA_MATERIAL = 0.08846153846153847

COURSE_UNIT_IDS = (
    (1, "GunPoint__impulse_v2", "family_member_backup_producer"),
    (2, "GunPointOldVersusYoung__impulse_v2", "beneficiary_strong"),
    (3, "PowerCons__impulse_v2", "conflict_unit_first_encounter"),
    (4, "Herring__impulse_v2", "heldout_only"),
    (5, "PowerCons__impulse_v2", "conflict_unit_re_encounter_mechanism_probe"),
    (6, "BirdChicken__burst_cls2", "out_of_family_control"),
)

PRE_REGISTERED = [
    {"id": "P1", "claim": "the Scope-v2 card matches GunPoint, GPOvY, "
                          "PowerCons and Herring, and does not match "
                          "BirdChicken"},
    {"id": "P2", "claim": "GunPoint and GPOvY convert the supplied candidate "
                          "through both gates"},
    {"id": "P3", "claim": "PowerCons #1 refuses the supplied candidate and "
                          "A5-adaptive emits exactly one narrowing PATCH "
                          "(card v1, content sha versioned)"},
    {"id": "P4", "claim": "Herring is either refused or already excluded by "
                          "v1; both are legal and which one is reported"},
    {"id": "P5", "claim": "PowerCons #2: A5-adaptive supplies nothing and is "
                          "refused nothing; K0-fixed supplies again and is "
                          "refused again"},
    {"id": "P6", "claim": "A5-adaptive saves >= 1 probe against K0-fixed and "
                          "its cumulative regret is non-inferior"},
    {"id": "P7", "claim": "harm events are zero in every arm"},
]


class Stop(Exception):
    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__("%s: %s" % (verdict, reason))
        self.verdict = verdict
        self.reason = reason


def _line_of(path: str, needle: str) -> str:
    text = (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines()
    line = next((i + 1 for i, t in enumerate(text) if needle in t), None)
    return "%s:%s" % (path, line)


def _units_by_id() -> dict[str, dict[str, Any]]:
    frozen = json.loads(sv.FREEZE_V4_JSON.read_text(encoding="utf-8"))
    return {str(row["unit_id"]): dict(row) for row in frozen["course"]}


def _course() -> list[dict[str, Any]]:
    by_id = _units_by_id()
    out = []
    for position, unit_id, role in COURSE_UNIT_IDS:
        unit = dict(by_id[unit_id])
        unit["position"] = position
        unit["role"] = role
        unit["slot"] = "%s#%d" % (unit_id.split("__")[0], position)
        unit["counts_toward_cumulative_regret"] = (
            position != REENCOUNTER_POSITION)
        out.append(unit)
    return out


def _features_of(unit: Mapping[str, Any]) -> dict[str, Any]:
    """The unit's binned Pattern view, off the production feature path.

    ``extract_public_features`` on the built cell is what the Fast path is
    handed at run time, so the offline tables below and the live rounds are
    reading the same thing.  The sealed oracle key is not opened for this.
    """
    cell = sv._half_cell(s1._build_cell(unit))
    block = np.asarray(cell["observation_block"], dtype=np.float64)
    raw = dict(extract_public_features(block, task_kind="classification"))
    return s1._binned_contract_leaves(raw)


# =========================================================================== #
# Part 0 gate -- the four fields, backfilled onto the recorded L1 replay
# =========================================================================== #
def _backfill_l1(card: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the four fields from L1 r1's own archived round records.

    L1 ran before the fields existed, which makes it the honest test: the
    values have to be recoverable from what was already written down, and the
    one row whose answer is known independently -- GPOvY, where a supplied
    candidate walked both gates -- has to come out right.
    """
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
        source_skill_of_candidate,
    )

    payload = json.loads(L1_R1.read_text(encoding="utf-8"))
    skill_id = str((payload.get("t1") or {}).get("card", {}).get("skill_id")
                   or "l1_ladder_v2_supply_v1")
    ast = payload["t1"]["card"]["observable_applicability"]
    table = []
    for row in payload.get("rows") or []:
        for record in row.get("rounds") or []:
            features = dict(record.get("fast_features_binned") or {})
            matched, _score = evaluate_applicability(ast, features)
            # Episodes are appended in probe order, one per legal Support
            # receipt, so the i-th Episode belongs to the i-th probe that was
            # not a verifier rejection (online_loop.py:406-428).
            probed = [probe for probe in (record.get("probes") or [])
                      if str(probe.get("kind")) == "probe"]
            episodes = list(record.get("episodes") or [])
            for index, episode in enumerate(episodes):
                candidate = (probed[index].get("candidate_id")
                             if index < len(probed) else None)
                table.append({
                    "position": row.get("position"),
                    "unit_id": row.get("unit_id"),
                    "episode_id": episode.get("episode_id"),
                    "candidate_id": candidate,
                    "source_skill_id": source_skill_of_candidate(candidate),
                    "source_skill_revision": (
                        ss.skill_content_sha(payload["t1"]["card"])
                        if source_skill_of_candidate(candidate) == skill_id
                        else None),
                    "scope_match": bool(matched),
                    "guidance_conditioned": skill_id in [
                        str(item) for item
                        in (record.get("retrieved_skill_ids") or ())],
                    "relation": episode.get("relation"),
                    "local_status": episode.get("local_status"),
                })
    beneficiary = next(
        (row for row in table
         if str(row["unit_id"]) == "GunPointOldVersusYoung__impulse_v2"), None)
    passed = bool(
        beneficiary
        and beneficiary["source_skill_id"] == skill_id
        and beneficiary["scope_match"] is True
        and beneficiary["guidance_conditioned"] is True
        and beneficiary["source_skill_revision"]
        and all(row["source_skill_id"] is None for row in table
                if row["unit_id"] != "GunPointOldVersusYoung__impulse_v2"))
    return {
        "check": "P0. the four fields backfill correctly on the L1 record",
        "pass": passed,
        "evidence": {
            "table": table,
            "gpovy_row": beneficiary,
            "card_content_sha": ss.skill_content_sha(payload["t1"]["card"]),
            "why_this_is_the_test": (
                "L1 ran before the fields existed, so every value here comes "
                "out of what was already recorded; the GPOvY row is the one "
                "whose answer is independently known"),
            "field_sites": {
                "source_skill_id": _line_of(
                    "methods/ttha/online_loop.py",
                    '"source_skill_id": str(source_skill_id)'),
                "source_skill_revision": _line_of(
                    "evaluation/functional/task_episode_harness/agentic/"
                    "source_skill.py", "def skill_content_sha"),
                "scope_match_by_skill_id": _line_of(
                    "evaluation/functional/run_e2_s1_curriculum_four_arms.py",
                    "def _scope_match_by_skill_id"),
                "guidance_conditioned_by_skill_id": _line_of(
                    "evaluation/functional/run_e2_s1_curriculum_four_arms.py",
                    "def _guidance_conditioned_by_skill_id"),
            },
        },
    }


# =========================================================================== #
# Part 0.5 -- Scope rule v2 and its match table
# =========================================================================== #
def _family_leaves() -> dict[str, Any] | None:
    return ss.s1a_pattern_family_leaves(TARGET_PROGRAM, audit_path=S1A_AUDIT)


def _pattern_family_provenance() -> str:
    return (
        "S1a cluster qualification: the Pattern intersection over the "
        "positives sharing one Program, computed by "
        "%s (helper %s) and frozen in %s at "
        "part_b.clusters[program=%s].pattern_intersection."
        % (_line_of("evaluation/functional/"
                    "run_e2_s1a_curriculum_oracle_audit.py",
                    "def _compatible_clusters"),
           _line_of("evaluation/functional/"
                    "run_e2_s1a_curriculum_oracle_audit.py",
                    "def _intersect_maps"),
           S1A_AUDIT.relative_to(PROJECT_ROOT).as_posix(),
           TARGET_PROGRAM))


def _seed_supply_rows() -> list[dict[str, Any]]:
    """The v4 record's GPMvF Episode, normalised for the compiler."""
    rows = [row for row in sv._v4_rows()
            if row["arm"] == sv.ARM_A5
            and int(row["position"]) <= sv.L1_BOUNDARY_POSITION]
    return sv._supply_rows_from(rows, card_installed_after=None)


def _compile_v0() -> dict[str, Any]:
    family = _family_leaves()
    if not family:
        raise Stop("PATTERN_FAMILY_UNDEFINED",
                   "S1a froze no qualifying cluster for %s, so there is no "
                   "mechanical family definition to use and inventing one "
                   "here is forbidden" % TARGET_PROGRAM)
    return ss.compile_supply_tier(
        _seed_supply_rows(), skill_id=SA1_SKILL_ID,
        legal_features=ss._edit_schema_features(PROJECT_ROOT),
        pattern_family=family,
        pattern_axis_provenance=_pattern_family_provenance())


def _match_table(ast: Mapping[str, Any], views: Mapping[str, Any]
                 ) -> list[dict[str, Any]]:
    out = []
    for position, unit_id, role in COURSE_UNIT_IDS:
        features = dict(views[unit_id])
        matched, score = evaluate_applicability(ast, features)
        out.append({"position": position, "unit_id": unit_id, "role": role,
                    "machine_match": bool(matched), "score": int(score)})
    return out


def _scope_v2_gate(views: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    compiled = _compile_v0()
    card = compiled["card"]
    if card is None:
        return ({"check": "P0.5 Scope v2 compiles the seed card",
                 "pass": False,
                 "evidence": {"withheld_because": compiled["withheld_because"],
                              "audit": compiled["audit"]}}, None)
    table = _match_table(card["observable_applicability"], views)
    matched = {row["unit_id"] for row in table if row["machine_match"]}
    expected = {"GunPoint__impulse_v2",
                "GunPointOldVersusYoung__impulse_v2",
                "PowerCons__impulse_v2", "Herring__impulse_v2"}
    leaves = card["observable_applicability"]["all"]
    features_named = [str(leaf["feature"]) for leaf in leaves]
    per_leaf = []
    for unit_id, view in views.items():
        misses = [leaf for leaf in leaves
                  if view.get(str(leaf["feature"])) != leaf["value"]]
        per_leaf.append({
            "unit_id": unit_id,
            "missed_leaves": [{"feature": leaf["feature"],
                               "card_value": leaf["value"],
                               "unit_value": view.get(str(leaf["feature"]))}
                              for leaf in misses]})
    return ({
        "check": "P0.5 the family-axis card matches the four impulse units "
                 "and not the burst control",
        "pass": bool(matched == expected
                     and len(features_named) == len(set(features_named))),
        "evidence": {
            "pattern_axis_kind": card["risk_guards"]["scope_v1"][
                "pattern_axis_kind"],
            "pattern_axis_provenance": card["risk_guards"]["scope_v1"][
                "pattern_axis_provenance"],
            "family_leaves": _family_leaves(),
            "scope_leaves": len(leaves),
            "distinct_features": len(set(features_named)),
            "duplicate_task_kind_leaf_gone": (
                features_named.count("task_kind") == 1),
            "match_table": table,
            "matched": sorted(matched),
            "expected": sorted(expected),
            "per_leaf_difference": per_leaf,
            "l1_v1_card_contrast": (
                "L1's n=1 card took the whole recorded view -- 18 leaves, 17 "
                "distinct features -- and matched 1 of 5 tail units; the "
                "incidental period_change_score leaf decided two of the three "
                "misses.  It is not in the family definition, so it is not on "
                "this card."),
            "card_content_sha": ss.skill_content_sha(card),
        },
    }, card)


# =========================================================================== #
# Part 1 -- the three write-backs, replayed offline
# =========================================================================== #
def _axes() -> frozenset[str]:
    return sr.contracted_axes(ss._edit_schema_features(PROJECT_ROOT))


def _install(base: Any, card: Mapping[str, Any], *, store_root: Path,
             tag: str) -> Any:
    snapshot, _applied = s1._apply_entries(base, [card],
                                           store_root=store_root, tag=tag)
    return snapshot


def _r1_row(*, unit_id: str, support_gain: float, delayed_gain: float,
            heldout_gain: float | None, card_sha: str) -> dict[str, Any]:
    return {
        "rule": "R1_positive",
        "unit_id": unit_id,
        "support_gain": support_gain,
        "delayed_gain": delayed_gain,
        "heldout_accuracy_gain": heldout_gain,
        "both_gates": "POSITIVE",
        "card_content_sha_when_earned": card_sha,
        "counts_toward_authorization": False,
        "counting_rule": (
            "earned while this card was in view, so it is "
            "Harness-conditioned and buys no tier and no wider Scope"),
    }


def _offline_replay(card: Mapping[str, Any], views: Mapping[str, Any],
                    store_root: Path) -> dict[str, Any]:
    """(a) R1 off L1's ledger, (b) R2/R3 off two historical negatives,
    (c) the narrowed card still admits every unit that did not refuse."""
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )

    axes = _axes()
    source_views = [dict(row.get("pattern_view") or {})
                    for row in card["risk_guards"]["evidence"]["sources"]]
    h0 = compile_snapshot(PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    base = _install(h0, card, store_root=store_root, tag="offline_v0")
    sha_v0 = ss.skill_content_sha(
        next(s for s in base.skills if str(s.skill_id) == SA1_SKILL_ID))
    chain = [{"version": "v0", "card_content_sha": sha_v0,
              "runtime_bundle_sha": base.runtime_bundle_sha,
              "produced_by": "compile_supply_tier (Scope rule v2)"}]
    steps: list[dict[str, Any]] = []

    # ---- (a) R1: the recorded L1 conversion, appended as one evidence row --
    l1 = json.loads(L1_R1.read_text(encoding="utf-8"))
    gpovy = next(row for row in l1["rows"]
                 if str(row["unit_id"]) == "GunPointOldVersusYoung__impulse_v2")
    episode = gpovy["rounds"][0]["episodes"][0]
    entry = next(s for s in base.skills if str(s.skill_id) == SA1_SKILL_ID)
    guards = sr.append_evidence_row(
        dict(entry.risk_guards or {}),
        _r1_row(unit_id=str(gpovy["unit_id"]),
                support_gain=float(episode["support_gain"]),
                delayed_gain=float(episode["delayed_gain"]),
                heldout_gain=float(gpovy["heldout_utility"]),
                card_sha=sha_v0))
    guards = sr.append_revision_log(guards, {
        "rule": "R1_positive", "trigger_unit": str(gpovy["unit_id"]),
        "surface": "risk_guards", "parent_card_content_sha": sha_v0,
        "source": L1_R1.relative_to(PROJECT_ROOT).as_posix()})
    r1 = sr.patch_card(base, skill_id=SA1_SKILL_ID, store_root=store_root,
                       tag="offline_r1", risk_guards=guards,
                       predicted_data_effect=("evidence_ledger_appended",))
    after_r1 = r1["snapshot"]
    ledger_after = dict(next(
        s for s in after_r1.skills
        if str(s.skill_id) == SA1_SKILL_ID).risk_guards)[sr.EVIDENCE_LEDGER_KEY]
    ast_after_r1 = json.loads(json.dumps(s1._plain(next(
        s for s in after_r1.skills
        if str(s.skill_id) == SA1_SKILL_ID).observable_applicability)))
    steps.append({
        "step": "(a) R1 -- L1's recorded conversion appends one ledger row",
        "pass": bool(len(ledger_after) == 1
                     and r1["card_sha"] != sha_v0
                     and ast_after_r1 == card["observable_applicability"]),
        "evidence": {
            "ledger_rows": ledger_after,
            "scope_unchanged": ast_after_r1 == card["observable_applicability"],
            "tier_unchanged": dict(next(
                s for s in after_r1.skills
                if str(s.skill_id) == SA1_SKILL_ID).risk_guards)["authority"],
            "receipts": r1["receipts"],
        },
    })
    chain.append({"version": "v0+r1", "card_content_sha": r1["card_sha"],
                  "runtime_bundle_sha": after_r1.runtime_bundle_sha,
                  "produced_by": "R1 evidence append (risk_guards PATCH)"})

    # ---- (b1) R2: the v3 PowerCons CONFLICT narrows the Scope -------------
    v3 = json.loads(V3_RUN.read_text(encoding="utf-8"))
    conflict = next(
        (episode
         for row in v3["rows"] if "PowerCons" in str(row["unit_id"])
         for record in (row.get("rounds") or [])
         for episode in (record.get("episodes") or [])
         if str(episode.get("relation")) == "CONFLICT"
         and str(episode.get("workflow_signature")) == TARGET_PROGRAM), None)
    if conflict is None:
        raise Stop("HISTORICAL_NEGATIVE_MISSING",
                   "no PowerCons CONFLICT on the hampel family in the v3 "
                   "artifact")
    refusing_view = dict(views["PowerCons__impulse_v2"])
    exclusion = sr.compile_exclusion(refusing_view=refusing_view,
                                     source_views=source_views, axes=axes)
    if not exclusion["leaves"]:
        raise Stop("R2_COMPILED_NOTHING", str(exclusion["empty_because"]))
    entry = next(s for s in after_r1.skills if str(s.skill_id) == SA1_SKILL_ID)
    narrowed = sr.narrow_applicability(
        json.loads(json.dumps(s1._plain(entry.observable_applicability))),
        exclusion["leaves"])
    guards = sr.append_revision_log(dict(entry.risk_guards or {}), {
        "rule": "R2_conflict", "trigger_unit": "PowerCons__impulse_v2",
        "trigger_reading": {"relation": "CONFLICT",
                            "support_gain": conflict.get("support_gain"),
                            "episode_id": conflict.get("episode_id")},
        "surface": "observable_applicability",
        "excluded": exclusion["leaves"],
        "parent_card_content_sha": r1["card_sha"],
        "source": V3_RUN.relative_to(PROJECT_ROOT).as_posix()})
    r2 = sr.patch_card(after_r1, skill_id=SA1_SKILL_ID, store_root=store_root,
                       tag="offline_r2", risk_guards=guards,
                       observable_applicability=narrowed,
                       predicted_data_effect=("scope_narrowed",))
    after_r2 = r2["snapshot"]
    chain.append({"version": "v1", "card_content_sha": r2["card_sha"],
                  "runtime_bundle_sha": after_r2.runtime_bundle_sha,
                  "produced_by": "R2 narrowing (observable_applicability "
                                 "PATCH) driven by the v3 PowerCons CONFLICT"})

    # ---- (b2) R3: the ECG200 outlier_mad harm reading demotes -------------
    ecg = json.loads(ECG200_HARM.read_text(encoding="utf-8"))
    harm_round = next(record for record in ecg["rounds"]
                      if int(record.get("harm_count") or 0) > 0)
    harm_episode = harm_round["episodes"][0]
    ecg_cell = cls._build_cell("ECG200", "fit_only_artifact")
    ecg_view = s1._binned_contract_leaves(dict(extract_public_features(
        np.asarray(ecg_cell["observation_block"], dtype=np.float64),
        task_kind="classification")))
    ecg_exclusion = sr.compile_exclusion(refusing_view=ecg_view,
                                         source_views=source_views, axes=axes)
    entry = next(s for s in after_r2.skills if str(s.skill_id) == SA1_SKILL_ID)
    guards = sr.append_demotion(dict(entry.risk_guards or {}), {
        "rule": "R3_negative_or_harm",
        "trigger_unit": "%s/%s" % (ecg["target"], ecg["condition"]),
        "program": str(harm_episode.get("workflow_signature")),
        "relation": str(harm_episode.get("relation")),
        "support_gain": harm_episode.get("support_gain"),
        "harm_count": int(harm_round.get("harm_count") or 0),
        "deprioritized_in": [dict(leaf) for leaf in ecg_exclusion["leaves"]],
        "note_kind": "structured_fields_only",
    })
    guards = sr.append_revision_log(guards, {
        "rule": "R3_negative_or_harm",
        "trigger_unit": "%s/%s" % (ecg["target"], ecg["condition"]),
        "surface": ("risk_guards+observable_applicability"
                    if ecg_exclusion["leaves"] else "risk_guards"),
        "excluded": ecg_exclusion["leaves"],
        "parent_card_content_sha": r2["card_sha"],
        "source": ECG200_HARM.relative_to(PROJECT_ROOT).as_posix()})
    entry_ast = json.loads(json.dumps(s1._plain(entry.observable_applicability)))
    r3_narrowed = (sr.narrow_applicability(entry_ast, ecg_exclusion["leaves"])
                   if ecg_exclusion["leaves"] else None)
    r3 = sr.patch_card(after_r2, skill_id=SA1_SKILL_ID, store_root=store_root,
                       tag="offline_r3", risk_guards=guards,
                       observable_applicability=r3_narrowed,
                       predicted_data_effect=("skill_demoted_in_domain",))
    after_r3 = r3["snapshot"]
    chain.append({"version": "v2", "card_content_sha": r3["card_sha"],
                  "runtime_bundle_sha": after_r3.runtime_bundle_sha,
                  "produced_by": "R3 demotion + exclusion driven by the "
                                 "ECG200 outlier_mad harm reading"})
    steps.append({
        "step": "(b) R2 and R3 -- two historical negatives each drive one "
                "revision, and every version is recoverable",
        "pass": bool(r2["card_sha"] != r1["card_sha"]
                     and r3["card_sha"] != r2["card_sha"]
                     and len({row["card_content_sha"] for row in chain})
                     == len(chain)),
        "evidence": {
            "r2_trigger": {"unit": "PowerCons__impulse_v2",
                           "relation": "CONFLICT",
                           "support_gain": conflict.get("support_gain"),
                           "artifact": V3_RUN.name},
            "r2_exclusion": exclusion,
            "r2_receipts": r2["receipts"],
            "r3_trigger": {"unit": "%s/%s" % (ecg["target"], ecg["condition"]),
                           "program": harm_episode.get("workflow_signature"),
                           "relation": harm_episode.get("relation"),
                           "harm_count": harm_round.get("harm_count"),
                           "artifact": ECG200_HARM.name},
            "r3_exclusion": ecg_exclusion,
            "r3_receipts": r3["receipts"],
            "version_chain": chain,
            "rollback": (
                "every parent bundle stays materialized and content-addressed "
                "in the fork store, so set_active(parent sha) restores the "
                "previous card bytes; the store refuses two different bodies "
                "under one sha"),
            "contracted_axes": sorted(axes),
        },
    })

    # ---- (c) no over-exclusion --------------------------------------------
    v1_entry = next(s for s in after_r2.skills
                    if str(s.skill_id) == SA1_SKILL_ID)
    v1_ast = json.loads(json.dumps(s1._plain(v1_entry.observable_applicability)))
    before = _match_table(card["observable_applicability"], views)
    after = _match_table(v1_ast, views)
    before_map = {row["unit_id"]: row["machine_match"] for row in before}
    after_map = {row["unit_id"]: row["machine_match"] for row in after}
    changed = sorted(uid for uid in before_map
                     if before_map[uid] != after_map[uid])
    steps.append({
        "step": "(c) the narrowed card excludes the refusing unit and "
                "nothing else",
        "pass": changed == ["PowerCons__impulse_v2"],
        "evidence": {
            "before": before_map, "after": after_map,
            "changed": changed,
            "expected_change": ["PowerCons__impulse_v2"],
            "bounded_loss_rule": (
                "the exclusion is the conjunction of the refusing unit's "
                "distinguishing values, so a unit differing on even one of "
                "them is untouched; what a narrowing costs can be bought back "
                "at the ladder price, what a predictive exclusion costs "
                "cannot"),
        },
    })
    return {"steps": steps, "pass": all(step["pass"] for step in steps),
            "version_chain": chain,
            "first_fault": next((step["step"] for step in steps
                                 if not step["pass"]), None)}


def run_gates() -> int:
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION + "/gates",
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
    }
    store_root = Path(tempfile.gettempdir()) / "sa1_gates"
    if store_root.exists():
        shutil.rmtree(store_root)
    try:
        views = {unit["unit_id"]: _features_of(unit)
                 for unit in {u["unit_id"]: u for u in _course()}.values()}
        payload["pattern_views"] = views
        p0 = _backfill_l1({})
        payload["part_0"] = p0
        if not p0["pass"]:
            raise Stop("PART_0_GATE_FAILED", p0["check"])
        p05, card = _scope_v2_gate(views)
        payload["part_0_5"] = p05
        payload["card_v0"] = card
        if not p05["pass"]:
            raise Stop("PART_0_5_GATE_FAILED", p05["check"])
        p1 = _offline_replay(card, views, store_root)
        payload["part_1"] = p1
        if not p1["pass"]:
            raise Stop("PART_1_GATE_FAILED", str(p1["first_fault"]))
        payload["verdict"] = {"verdict": "SA1_OFFLINE_GATES_PASS",
                              "first_fault": None}
    except Stop as stop:
        payload["verdict"] = {"verdict": stop.verdict,
                              "first_fault": stop.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        payload["verdict"] = {"verdict": "INSTRUMENT_UNREADABLE",
                              "first_fault": "%s: %s" % (type(exc).__name__,
                                                         exc),
                              "traceback": traceback.format_exc()}
    payload["ledger"] = {"llm": 0, "fit": 0,
                         "wall_seconds": round(time.time() - started, 1),
                         "downloads": 0}
    GATES_JSON.write_text(json.dumps(s1._plain(payload), ensure_ascii=False,
                                     indent=1, sort_keys=True, default=str)
                          + "\n", encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"],
                      "artifact": str(GATES_JSON)}, ensure_ascii=False,
                     indent=1))
    return 0 if payload["verdict"]["verdict"] == "SA1_OFFLINE_GATES_PASS" else 1


# =========================================================================== #
# freeze
# =========================================================================== #
def freeze() -> int:
    gates = json.loads(GATES_JSON.read_text(encoding="utf-8"))
    if (gates.get("verdict") or {}).get("verdict") != "SA1_OFFLINE_GATES_PASS":
        print("REFUSING to freeze: offline gates did not pass")
        return 1
    payload = {
        "protocol_version": PROTOCOL_VERSION + "/freeze",
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "course": _course(),
        "arms": [
            {"arm": ARM_A3, "label": "A3-reset",
             "base": "h0", "card": None, "revision": False},
            {"arm": ARM_K0, "label": "K0-fixed",
             "base": "K0 + Scope-v2 seed card v0", "card": SA1_SKILL_ID,
             "revision": False},
            {"arm": ARM_A5, "label": "A5-adaptive",
             "base": "K0 + Scope-v2 seed card v0", "card": SA1_SKILL_ID,
             "revision": True, "rules": ["R1", "R2", "R3"]},
        ],
        "arm_symmetry": (
            "K0-fixed and A5-adaptive start every unit from their own current "
            "snapshot and carry no Episodes, so the only difference between "
            "them is whether the card may be revised.  A3-reset carries no "
            "card at all."),
        "card_seed": {
            "skill_id": SA1_SKILL_ID,
            "evidence": "the v4 record's GunPointMaleVersusFemale Episode, "
                        "one strong unguided positive (ladder v2 price)",
            "authority": "supplies_candidates only; no execution, no deploy",
            "scope_rule": "v2 five bearing axes, Pattern axis = S1a family",
            "content_sha_v0": (gates["part_0_5"]["evidence"]
                               ["card_content_sha"]),
        },
        "reencounter_rule": (
            "position %d is PowerCons met a second time.  Its readings are a "
            "mechanism probe and are reported in their own row; cumulative "
            "regret on the headline counts distinct units only, so meeting "
            "one domain twice cannot double-count against any arm."
            % REENCOUNTER_POSITION),
        "scoring": {
            "primary": "ITT -- a Scope-qualified unit whose supplied "
                       "candidate did not reach the pool is an A5 failure",
            "delta_material": DELTA_MATERIAL,
            "harm_bar": cls.HARM_BAR,
            "headline": "probes saved + refusals avoided (A5-adaptive vs "
                        "K0-fixed) and the cumulative regret difference",
        },
        "pre_registered_predictions": PRE_REGISTERED,
        "stop_rules": [
            "PowerCons #1 converts unexpectedly -> the mechanism readout is "
            "carried by Herring's refusal instead and this is reported, not "
            "repaired",
            "a refusal happens and R2 does not fire -> stop, single "
            "postmortem, no course change, no threshold change, no r2",
        ],
        "budget": {"llm": LLM_CAP, "fit": FIT_CAP,
                   "wall_seconds": WALL_SECONDS_CAP, "downloads": 0},
        "single_run_wording": (
            "one run records SA1_DEVELOPMENT_SIGNAL at most; compound wording "
            "still requires a sampling replicate"),
    }
    FREEZE_JSON.write_text(json.dumps(s1._plain(payload), ensure_ascii=False,
                                      indent=1, sort_keys=True, default=str)
                           + "\n", encoding="utf-8")
    print("frozen: %s" % FREEZE_JSON)
    return 0


# =========================================================================== #
# Part 2 -- the live course
# =========================================================================== #
def _supplied_episodes(scored: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [episode for record in (scored.get("rounds") or [])
            for episode in (record.get("episodes") or [])
            if str(episode.get("source_skill_id") or "") == SA1_SKILL_ID]


def _card_readout(scored: Mapping[str, Any]) -> dict[str, Any]:
    rounds = scored.get("rounds") or []
    supplied = _supplied_episodes(scored)
    scope_match = any(
        bool((record.get("scope_match_by_skill_id") or {}).get(SA1_SKILL_ID))
        for record in rounds)
    in_pool = sum(1 for record in rounds
                  for candidate in (record.get("pool") or [])
                  if str(candidate) == "cand_skill_%s" % SA1_SKILL_ID)
    verifier_rejected = sum(
        1 for record in rounds for probe in (record.get("probes") or [])
        if str(probe.get("candidate_id")) == "cand_skill_%s" % SA1_SKILL_ID
        and str(probe.get("kind")) == "verifier_rejected")
    converted = [episode for episode in supplied
                 if str(episode.get("relation")) == "POSITIVE"
                 and str(episode.get("local_status")) == "LOCAL_ACTIVE"]
    refused = [episode for episode in supplied if episode not in converted]
    return {
        "scope_match": scope_match,
        "supplied_in_pool": in_pool,
        "supplied_probed": len(supplied),
        "supplied_verifier_rejected": verifier_rejected,
        "converted": len(converted),
        "refused": len(refused),
        "refusal_relations": [str(e.get("relation")) for e in refused],
        "converted_episodes": converted,
        "refused_episodes": refused,
    }


def _revise(snapshot: Any, scored: Mapping[str, Any],
            unit: Mapping[str, Any], *, store_root: Path,
            card_sha: str) -> dict[str, Any]:
    """One boundary's worth of revision for A5-adaptive.

    R1 when both gates approved the supplied candidate, R2 when the Scope
    admitted the unit and the Target refused it, R3 when the refusal was
    NEGATIVE or the unit produced a harm event.  Nothing else is written, and
    a unit the Scope never admitted produces no revision at all.
    """
    entry = next((s for s in snapshot.skills
                  if str(s.skill_id) == SA1_SKILL_ID), None)
    if entry is None:
        return {"applied": [], "snapshot": snapshot, "card_sha": card_sha}
    read = _card_readout(scored)
    guards = dict(entry.risk_guards or {})
    ast = json.loads(json.dumps(s1._plain(entry.observable_applicability)))
    applied: list[str] = []
    narrowed = None
    exclusion: dict[str, Any] | None = None

    for episode in read["converted_episodes"]:
        guards = sr.append_evidence_row(guards, _r1_row(
            unit_id=str(unit["unit_id"]),
            support_gain=float(episode.get("support_gain") or 0.0),
            delayed_gain=float(episode.get("delayed_gain") or 0.0),
            heldout_gain=float(scored.get("heldout_utility") or 0.0),
            card_sha=card_sha))
        guards = sr.append_revision_log(guards, {
            "rule": "R1_positive", "trigger_unit": str(unit["unit_id"]),
            "surface": "risk_guards", "parent_card_content_sha": card_sha})
        applied.append("R1")

    negative = [e for e in read["refused_episodes"]
                if str(e.get("relation")) == "NEGATIVE"]
    harmed = bool(scored.get("harm_event"))
    if read["refused"] or harmed:
        source_views = [dict(row.get("pattern_view") or {}) for row in
                        (guards.get("evidence") or {}).get("sources") or []]
        refusing_view = dict(
            (scored.get("rounds") or [{}])[0].get("fast_features_binned") or {})
        exclusion = sr.compile_exclusion(refusing_view=refusing_view,
                                         source_views=source_views,
                                         axes=_axes())
        rule = "R3_negative_or_harm" if (negative or harmed) else "R2_conflict"
        if exclusion["leaves"]:
            narrowed = sr.narrow_applicability(ast, exclusion["leaves"])
            applied.append("R2" if rule == "R2_conflict" else "R3")
        if rule == "R3_negative_or_harm":
            guards = sr.append_demotion(guards, {
                "rule": rule, "trigger_unit": str(unit["unit_id"]),
                "relation": [str(e.get("relation")) for e in negative],
                "harm_event": harmed,
                "deprioritized_in": exclusion["leaves"],
                "note_kind": "structured_fields_only"})
            if "R3" not in applied:
                applied.append("R3")
        guards = sr.append_revision_log(guards, {
            "rule": rule, "trigger_unit": str(unit["unit_id"]),
            "surface": ("risk_guards+observable_applicability"
                        if exclusion["leaves"] else "risk_guards"),
            "excluded": exclusion["leaves"],
            "empty_because": exclusion["empty_because"],
            "parent_card_content_sha": card_sha})

    if not applied and narrowed is None:
        return {"applied": [], "snapshot": snapshot, "card_sha": card_sha,
                "readout": read, "exclusion": exclusion}
    patched = sr.patch_card(
        snapshot, skill_id=SA1_SKILL_ID, store_root=store_root,
        tag="revise_%d" % int(unit["position"]),
        risk_guards=guards, observable_applicability=narrowed,
        predicted_data_effect=tuple(applied) or ("skill_revised",))
    return {"applied": applied, "snapshot": patched["snapshot"],
            "card_sha": patched["card_sha"], "receipts": patched["receipts"],
            "readout": read, "exclusion": exclusion}


def run_course(seed: str = "r1", *, resume: bool = False,
               finalize: bool = False) -> int:
    import run_e2_ps0c_ps1 as ps0c

    gates = json.loads(GATES_JSON.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
    course = list(frozen["course"])
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "seed": seed,
        "replicate_kind": "sampling",
        "course": course,
        "arms": frozen["arms"],
        "card_seed": frozen["card_seed"],
        "reencounter_rule": frozen["reencounter_rule"],
        "scoring": frozen["scoring"],
        "pre_registered_predictions": PRE_REGISTERED,
        "gates": {"part_0": gates["part_0"], "part_0_5": gates["part_0_5"],
                  "part_1": gates["part_1"],
                  "verdict": gates["verdict"]},
    }
    rows: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = []
    ledger = {"llm": 0, "fit": 0}
    done: set[tuple[int, str]] = set()
    if (resume or finalize) and CHECKPOINT.is_file():
        saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        rows = list(saved.get("rows") or [])
        revisions = list(saved.get("revisions") or [])
        chain = list(saved.get("version_chain") or [])
        ledger = {"llm": int((saved.get("ledger") or {}).get("llm") or 0),
                  "fit": int((saved.get("ledger") or {}).get("fit") or 0)}
        started = time.time() - float(saved.get("wall_seconds") or 0.0)
        done = {(int(row["position"]), str(row["arm"])) for row in rows}
        payload["resumed_from_checkpoint"] = sorted(
            "%s/%s" % (pos, arm) for pos, arm in done)

    def _save() -> None:
        CHECKPOINT.write_text(json.dumps(ps0c.redact(s1._plain({
            "rows": rows, "revisions": revisions, "version_chain": chain,
            "ledger": ledger,
            "wall_seconds": round(time.time() - started, 1),
        })), indent=1, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8")

    stopped: str | None = None
    if finalize:
        return _finish(payload, rows, revisions, chain, ledger,
                       stopped=None, started=started, ps0c=ps0c)
    try:
        probe = ps0c.probe_new_backend()
        payload["backend_probe"] = ps0c.redact(probe)
        print("PROBE ok=%s model=%s" % (probe.get("ok"),
                                        probe.get("returned_model")),
              flush=True)
        if not probe.get("ok"):
            raise Stop("BACKEND_UNAVAILABLE", str(probe.get("reason")))

        store_root = Path(tempfile.gettempdir()) / ("sa1_%s" % seed)
        if not resume and store_root.exists():
            shutil.rmtree(store_root)
        k0 = s1.compile_k0(store_root)
        card = gates["card_v0"]
        k0_fixed = _install(k0["k0"], card, store_root=store_root / "seed",
                            tag="k0_fixed")
        a5 = _install(k0["k0"], card, store_root=store_root / "seed",
                      tag="a5_v0")
        card_sha = ss.skill_content_sha(
            next(s for s in a5.skills if str(s.skill_id) == SA1_SKILL_ID))
        if not chain:
            chain = [{"version": "v0", "card_content_sha": card_sha,
                      "installed_before_position": 1,
                      "runtime_bundle_sha": a5.runtime_bundle_sha}]
        payload["k0"] = {"h0_sha": k0["h0_sha"], "k0_sha": k0["k0_sha"],
                         "purity": k0["purity"],
                         "k0_fixed_sha": k0_fixed.runtime_bundle_sha,
                         "a5_v0_sha": a5.runtime_bundle_sha}
        backend = cls._live_backend(LLM_CAP)

        for unit in course:
            position = int(unit["position"])
            uid = str(unit["unit_id"])
            if time.time() - started > WALL_SECONDS_CAP:
                raise Stop("COMPUTE_BUDGET_EXCEEDED", "wall cap before " + uid)
            if ledger["llm"] >= LLM_CAP or ledger["fit"] >= FIT_CAP:
                raise Stop("COMPUTE_BUDGET_EXCEEDED", "budget before " + uid)
            print("UNIT %d %s (%s)" % (position, uid, unit["role"]),
                  flush=True)
            cell = sv._half_cell(s1._build_cell(unit))

            for arm in ARMS:
                if (position, arm) in done:
                    continue
                base = {ARM_A3: k0["h0"], ARM_K0: k0_fixed, ARM_A5: a5}[arm]
                result = s1.run_unit(
                    unit=unit, cell=cell, arm=arm, base_snapshot=base,
                    carried_episodes=(), agent_factory=cls._live_agent,
                    backend=backend, store_root=store_root,
                    rounds=HALF_ROUNDS, fit_cap=FIT_PER_UNIT_PER_ARM,
                    carried_stamps={})
                ledger["llm"] = int(backend.calls)
                ledger["fit"] += int(result.get("consumer_fits") or 0)
                scored = sv._score_unit(unit, arm, result)
                scored["slot"] = unit["slot"]
                scored["counts_toward_cumulative_regret"] = bool(
                    unit["counts_toward_cumulative_regret"])
                scored["card"] = _card_readout(scored)
                scored["card_content_sha_in_view"] = (
                    card_sha if arm == ARM_A5 else
                    (chain[0]["card_content_sha"] if arm == ARM_K0 else None))
                rows.append(scored)
                _save()
                print("  %-10s deploy=%-30s regret=%+.4f probes=%d "
                      "supplied=%d converted=%d refused=%d"
                      % (ARM_LABEL[arm], scored["deploy_source"],
                         scored["regret"], scored["probes"],
                         scored["card"]["supplied_in_pool"],
                         scored["card"]["converted"],
                         scored["card"]["refused"]), flush=True)

                if arm == ARM_A5:
                    revision = _revise(a5, scored, unit,
                                       store_root=store_root / "revise",
                                       card_sha=card_sha)
                    a5 = revision["snapshot"]
                    if revision["applied"]:
                        chain.append({
                            "version": "v%d" % len(chain),
                            "card_content_sha": revision["card_sha"],
                            "produced_by": "+".join(revision["applied"]),
                            "trigger_unit": uid,
                            "after_position": position,
                            "runtime_bundle_sha": a5.runtime_bundle_sha})
                        print("    REVISION %s -> %s"
                              % ("+".join(revision["applied"]),
                                 revision["card_sha"][:12]), flush=True)
                    card_sha = revision["card_sha"]
                    revisions.append({
                        "position": position, "unit_id": uid,
                        "applied": revision["applied"],
                        "readout": revision.get("readout"),
                        "exclusion": revision.get("exclusion"),
                        "receipts": revision.get("receipts"),
                        "card_content_sha_after": revision["card_sha"]})
                    _save()

                    read = scored["card"]
                    if read["refused"] and not any(
                            rule in revision["applied"]
                            for rule in ("R2", "R3")):
                        raise Stop(
                            "SA1_WRITE_BACK_NOT_TRIGGERED",
                            "%s refused the supplied candidate (%s) and no "
                            "narrowing was written"
                            % (uid, read["refusal_relations"]))
    except Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {"verdict": stopped,
                           "reason": ps0c.redact("%s: %s"
                                                 % (type(exc).__name__, exc)),
                           "traceback": ps0c.redact(traceback.format_exc())}
    return _finish(payload, rows, revisions, chain, ledger,
                   stopped=stopped, started=started, ps0c=ps0c)


# =========================================================================== #
# scoring and rendering
# =========================================================================== #
def _summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        distinct = [row for row in arm_rows
                    if row.get("counts_toward_cumulative_regret", True)]
        out[arm] = {
            "label": ARM_LABEL[arm],
            "units": len(arm_rows),
            "distinct_units": len(distinct),
            "cumulative_regret_distinct_units": sum(row["regret"]
                                                    for row in distinct),
            "probes": sum(row["probes"] for row in arm_rows),
            "probes_distinct_units": sum(row["probes"] for row in distinct),
            "supplied_in_pool": sum((row.get("card") or {}).get(
                "supplied_in_pool", 0) for row in arm_rows),
            "supplied_refused": sum((row.get("card") or {}).get("refused", 0)
                                    for row in arm_rows),
            "supplied_converted": sum((row.get("card") or {}).get(
                "converted", 0) for row in arm_rows),
            "harm_events": sum(1 for row in arm_rows if row["harm_event"]),
            "worst_class_min": min((row["worst_class_delta"]
                                    for row in arm_rows), default=0.0),
            "llm": sum(row["llm_calls"] for row in arm_rows),
            "consumer_fits": sum(row["consumer_fits"] for row in arm_rows),
        }
    return out


def _headline(summary: Mapping[str, Any],
              rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a5 = summary.get(ARM_A5) or {}
    k0 = summary.get(ARM_K0) or {}
    probes_saved = int(k0.get("probes", 0)) - int(a5.get("probes", 0))
    refusals_avoided = (int(k0.get("supplied_refused", 0))
                        - int(a5.get("supplied_refused", 0)))
    regret_gap = (float(k0.get("cumulative_regret_distinct_units", 0.0))
                  - float(a5.get("cumulative_regret_distinct_units", 0.0)))
    reencounter = {
        ARM_LABEL[row["arm"]]: {
            "supplied_in_pool": (row.get("card") or {}).get("supplied_in_pool"),
            "refused": (row.get("card") or {}).get("refused"),
            "converted": (row.get("card") or {}).get("converted"),
            "probes": row["probes"], "regret": row["regret"]}
        for row in rows if int(row["position"]) == REENCOUNTER_POSITION}
    moved = bool(probes_saved >= 1 or refusals_avoided >= 1)
    return {
        "core_positive_effect_moved": moved,
        "probes_saved_a5_vs_k0": probes_saved,
        "refusals_avoided_a5_vs_k0": refusals_avoided,
        "regret_gap_a5_vs_k0_distinct_units": regret_gap,
        "regret_non_inferior": regret_gap >= 0.0,
        "harm_all_zero": all(int((summary.get(arm) or {}).get(
            "harm_events", 0)) == 0 for arm in ARMS),
        "reencounter_readout": reencounter,
    }


def _prediction_table(rows, revisions, chain, gates) -> list[dict[str, Any]]:
    by_pos = {(int(row["position"]), str(row["arm"])): row for row in rows}

    def card(position: int, arm: str) -> dict[str, Any]:
        return (by_pos.get((position, arm)) or {}).get("card") or {}

    out = []
    p05 = (gates.get("part_0_5") or {}).get("evidence") or {}
    out.append({"id": "P1", "held": bool(
        p05.get("matched") == p05.get("expected")),
        "observed": "matched %s" % (p05.get("matched"),)})

    converted = [pos for pos in (1, 2) if card(pos, ARM_A5).get("converted")]
    out.append({"id": "P2", "held": converted == [1, 2],
                "observed": "A5-adaptive conversions at positions %s"
                            % (converted,)})

    narrowings = [row for row in revisions
                  if "R2" in (row.get("applied") or [])
                  or "R3" in (row.get("applied") or [])]
    first = [row for row in narrowings if int(row["position"]) == 3]
    out.append({"id": "P3",
                "held": bool(card(3, ARM_A5).get("refused")
                             and len(first) == 1
                             and len(chain) >= 2),
                "observed": "PowerCons#1 refused=%s, narrowing PATCHes at "
                            "position 3 = %d, version chain = %s"
                            % (card(3, ARM_A5).get("refused"), len(first),
                               [row["version"] for row in chain])})

    herring = card(4, ARM_A5)
    out.append({"id": "P4",
                "held": bool(herring.get("refused")
                             or not herring.get("scope_match")),
                "observed": "Herring scope_match=%s supplied=%s refused=%s"
                            % (herring.get("scope_match"),
                               herring.get("supplied_in_pool"),
                               herring.get("refused"))})

    a5_re, k0_re = card(5, ARM_A5), card(5, ARM_K0)
    out.append({"id": "P5",
                "held": bool(a5_re.get("supplied_in_pool") == 0
                             and a5_re.get("refused") == 0
                             and k0_re.get("supplied_in_pool", 0) >= 1
                             and k0_re.get("refused", 0) >= 1),
                "observed": "A5 supplied=%s refused=%s | K0 supplied=%s "
                            "refused=%s"
                            % (a5_re.get("supplied_in_pool"),
                               a5_re.get("refused"),
                               k0_re.get("supplied_in_pool"),
                               k0_re.get("refused"))})
    return out


def _verdict(headline, predictions, rows, revisions, *, stopped):
    if stopped:
        return {"verdict": stopped,
                "reason": "stopped before the course completed; readings "
                          "below are partial and the stop rule forbids a "
                          "second run"}
    mechanism = bool(headline["probes_saved_a5_vs_k0"] >= 1
                     or headline["refusals_avoided_a5_vs_k0"] >= 1)
    if (mechanism and headline["regret_non_inferior"]
            and headline["harm_all_zero"]
            and any(row.get("applied") for row in revisions)):
        return {
            "verdict": "SA1_DEVELOPMENT_SIGNAL",
            "reason": (
                "a refusal produced a structured narrowing, the narrowed card "
                "stopped supplying the domain that refused it while every "
                "other domain kept its candidate, and the arm that revises "
                "beat the arm that cannot on the mechanism readout with "
                "non-inferior regret and zero harm"),
            "single_run_wording": (
                "one run; compound wording requires a sampling replicate"),
        }
    return {
        "verdict": "SA1_NO_MECHANISM_DIFFERENCE",
        "reason": ("the revision loop ran but produced no material mechanism "
                   "difference against the frozen-card arm"),
    }


def _finish(payload, rows, revisions, chain, ledger, *, stopped, started,
            ps0c) -> int:
    payload["rows"] = rows
    payload["revisions"] = revisions
    payload["card_version_chain"] = chain
    payload["summary"] = _summarise(rows)
    payload["headline"] = _headline(payload["summary"], rows)
    payload["prediction_table"] = _prediction_table(
        rows, revisions, chain, payload.get("gates") or {})
    payload["verdict"] = _verdict(payload["headline"],
                                  payload["prediction_table"], rows,
                                  revisions, stopped=stopped)
    payload["ledger"] = {
        "llm": ledger["llm"], "llm_cap": LLM_CAP,
        "fit": ledger["fit"], "fit_cap": FIT_CAP,
        "wall_seconds": round(time.time() - started, 1),
        "wall_seconds_cap": WALL_SECONDS_CAP, "downloads": 0,
    }
    payload["obligations"] = _obligations()
    OUT_JSON.write_text(json.dumps(ps0c.redact(s1._plain(payload)),
                                   ensure_ascii=False, indent=1,
                                   sort_keys=True, default=str) + "\n",
                        encoding="utf-8")
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"],
                      "headline": payload["headline"],
                      "ledger": payload["ledger"]},
                     ensure_ascii=False, indent=1, default=str))
    return 0


def _obligations() -> dict[str, Any]:
    return {
        "course_frozen_before_any_live_run": True,
        "thresholds_and_authorization_unmodified": (
            "MATERIAL, the TRY tier leave-one-out, the RISK tier, the supply "
            "tier count, the execution and deployment gates and the "
            "prompt/model/budget protocol are untouched"),
        "fault_routes_and_router_unmodified": True,
        "minipipe_fault_routing_not_touched": True,
        "sealed_material_not_opened": (
            "Epilepsy2 and the s1_oracle keys were not read; every Pattern "
            "view in this book comes from extract_public_features on the "
            "built cell, which is the production path"),
        "new_data_units_operators_consumers": 0,
        "downloads": 0,
        "full_repo_pytest_not_run": True,
        "subagents_spawned": 0,
        "q1_residue": (
            "the applicability surface is authorized by RETRIEVAL_MISS alone, "
            "a cause named for the widening direction; SA-1 used it as the "
            "token for a narrowing PATCH rather than mint a code, and this "
            "stays open as Q1"),
        "single_run": True,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    head = payload["headline"]
    summary = payload["summary"]
    lines = [
        "# SA-1 minimal r1 -- Skill as an updatable hypothesis",
        "",
        "**核心正效果移动:%s + probe 省 %d / 避免挨拒 %d / regret 差 %+.4f**"
        % ("是" if head["core_positive_effect_moved"] else "否",
           head["probes_saved_a5_vs_k0"], head["refusals_avoided_a5_vs_k0"],
           head["regret_gap_a5_vs_k0_distinct_units"]),
        "",
        "判词 **%s** -- %s" % (payload["verdict"]["verdict"],
                               payload["verdict"].get("reason")),
        "",
        "## Offline gates",
        "",
        "| part | check | pass |",
        "|---|---|---|",
    ]
    gates = payload.get("gates") or {}
    for key in ("part_0", "part_0_5"):
        row = gates.get(key) or {}
        lines.append("| %s | %s | %s |" % (key, row.get("check"),
                                           row.get("pass")))
    for step in (gates.get("part_1") or {}).get("steps") or []:
        lines.append("| part_1 | %s | %s |" % (step["step"], step["pass"]))

    lines += ["", "## Card version chain", "",
              " -> ".join("%s `%s`" % (row["version"],
                                       str(row["card_content_sha"])[:12])
                          for row in payload["card_version_chain"]) or "-",
              "", "## Per unit, per arm", "",
              "| # | unit | arm | deploy | regret | probes | supplied | "
              "converted | refused | harm |", "|---|---|---|---|---|---|---|"
              "---|---|---|"]
    for row in sorted(payload["rows"], key=lambda r: (int(r["position"]),
                                                      str(r["arm"]))):
        card = row.get("card") or {}
        lines.append("| %s%s | %s | %s | %s | %+.4f | %d | %s | %s | %s | %s |"
                     % (row["position"],
                        "*" if not row.get(
                            "counts_toward_cumulative_regret", True) else "",
                        row["unit_id"].split("__")[0],
                        ARM_LABEL.get(row["arm"], row["arm"]),
                        row.get("deploy_source"),
                        row["regret"], row["probes"],
                        card.get("supplied_in_pool"), card.get("converted"),
                        card.get("refused"), row["harm_event"]))
    lines += ["", "`*` = the re-encounter slot; a mechanism readout that does "
              "not count toward cumulative regret.", "",
              "## Arm totals (distinct units only for regret)", "",
              "| arm | regret | probes | supplied | converted | refused | "
              "harm | llm | fit |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        row = summary.get(arm) or {}
        lines.append("| %s | %+.4f | %d | %d | %d | %d | %d | %d | %d |"
                     % (row.get("label"),
                        row.get("cumulative_regret_distinct_units", 0.0),
                        row.get("probes", 0), row.get("supplied_in_pool", 0),
                        row.get("supplied_converted", 0),
                        row.get("supplied_refused", 0),
                        row.get("harm_events", 0), row.get("llm", 0),
                        row.get("consumer_fits", 0)))

    lines += ["", "## Pre-registered predictions", "",
              "| id | claim | held | observed |", "|---|---|---|---|"]
    claims = {row["id"]: row["claim"] for row in PRE_REGISTERED}
    for row in payload["prediction_table"]:
        lines.append("| %s | %s | %s | %s |"
                     % (row["id"], claims.get(row["id"], ""),
                        "yes" if row["held"] else "**no**", row["observed"]))
    p6 = (head["probes_saved_a5_vs_k0"] >= 1) and head["regret_non_inferior"]
    lines.append("| P6 | %s | %s | probes saved %d, regret gap %+.4f |"
                 % (claims["P6"], "yes" if p6 else "**no**",
                    head["probes_saved_a5_vs_k0"],
                    head["regret_gap_a5_vs_k0_distinct_units"]))
    lines.append("| P7 | %s | %s | harm events %s |"
                 % (claims["P7"], "yes" if head["harm_all_zero"] else "**no**",
                    {ARM_LABEL[arm]: (summary.get(arm) or {}).get(
                        "harm_events") for arm in ARMS}))

    lines += ["", "## Revisions", "",
              "| after # | unit | rules | excluded | card sha |",
              "|---|---|---|---|---|"]
    for row in payload["revisions"]:
        exclusion = row.get("exclusion") or {}
        lines.append("| %s | %s | %s | %s | `%s` |"
                     % (row["position"], row["unit_id"].split("__")[0],
                        "+".join(row.get("applied") or []) or "-",
                        ", ".join("%s==%s" % (leaf["feature"], leaf["value"])
                                  for leaf in exclusion.get("leaves") or [])
                        or "-",
                        str(row.get("card_content_sha_after"))[:12]))

    ledger = payload["ledger"]
    lines += ["", "## Cost", "",
              "LLM %d/%d, consumer fits %d/%d, wall %.0f s / %d s, downloads "
              "%d." % (ledger["llm"], ledger["llm_cap"], ledger["fit"],
                       ledger["fit_cap"], ledger["wall_seconds"],
                       ledger["wall_seconds_cap"], ledger["downloads"]),
              "", "## Obligations", ""]
    for key, value in sorted(payload["obligations"].items()):
        lines.append("- **%s**: %s" % (key, value))
    if payload.get("stop"):
        lines += ["", "## Stop", "", "`%s` -- %s"
                  % (payload["stop"]["verdict"], payload["stop"]["reason"])]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--seed", default="r1")
    args = parser.parse_args(argv)
    if args.gates:
        return run_gates()
    if args.freeze:
        return freeze()
    if args.run or args.resume or args.finalize:
        return run_course(args.seed, resume=args.resume,
                          finalize=args.finalize)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
