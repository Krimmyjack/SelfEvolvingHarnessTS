"""S1-v2 -- the forward evolution course, run twice.

Design: ``docs/S1V2_DESIGN_DRAFT_2026-08-27.md``.  This is the Skill/Memory
compounding exam: one Harness walks a fixed sequence of units, and the
question is whether A5-online's cross-unit write-back buys anything a
per-unit reset does not.

Everything the course needs already exists and is not re-litigated here:

* the reader -- ``fast_agent._supply_rung_candidates`` (W-1);
* the producer -- ``source_skill.compile_supply_tier`` (P0), two independent
  unguided positives, ``supplies_candidates`` only;
* the guard channel -- ``runner.run_risk_skill_lifecycle`` (B);
* the feedback surface -- the M-1 half protocol: one held-in round whose
  Support is ``concat(r1_support, r2_support)`` and whose delayed is
  ``concat(r1_delayed, r2_delayed)``, so the dual gate is preserved and the
  confirmation surface doubles.

Part 0 (``--freeze``) is arithmetic only: recompute every unit's margin under
the half protocol from the ps0b sealed counts, pick the course mechanically,
and refuse to spend a single LLM call unless the treatment group provably
exists.

  python evaluation/functional/run_e2_s1v2_forward_course.py --freeze
  python evaluation/functional/run_e2_s1v2_forward_course.py --run --seed r1
  python evaluation/functional/run_e2_s1v2_forward_course.py --resume --seed r1
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

import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    evaluate_applicability,
)
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    source_skill as ss,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
ORACLE_DIR = E2 / "s1_oracle"
PS0B_JSON = E2 / "ps0b_confirmation_surface_audit.json"
FREEZE_JSON = E2 / "s1v2_course_freeze.json"
FREEZE_MD = E2 / "s1v2_course_freeze.md"
# r2: the main line released the two exam units as beneficiaries and redefined
# the regret gate.  The r1 stop artifact is kept as the record of why.
FREEZE_R2_JSON = E2 / "s1v2_course_freeze_r2.json"
FREEZE_R2_MD = E2 / "s1v2_course_freeze_r2.md"
# v3: arbitration A.  Only the producer side changes -- they are now picked on
# demonstrated cold-discovery rate rather than on sealed margin alone, which
# is what run 1 showed was the missing precondition.
FREEZE_V3_JSON = E2 / "s1v2_course_freeze_v3.json"
FREEZE_V3_MD = E2 / "s1v2_course_freeze_v3.md"
# v4: the final throw.  Producer selection gains a third criterion (live
# Support pass rate) and a backup producer, and the weak beneficiary becomes
# the unit whose live Support is honestly thin.
FREEZE_V4_JSON = E2 / "s1v2_course_freeze_v4.json"
FREEZE_V4_MD = E2 / "s1v2_course_freeze_v4.md"

PROTOCOL_VERSION = "s1v2_forward_course_v1"
EVIDENCE_GRADE = "development"

TARGET_OPERATOR = "hampel_filter"
MARGIN_BAR = 2.0

# Frozen exclusions, each with the book it is spent on.
EXCLUDED = {
    "GunPointAgeSpan__impulse_v2": "dual-source A (PS-0)",
    "PowerCons__impulse_v2": "dual-source B (PS-0c)",
    "GunPointOldVersusYoung__impulse_v2": "PS-2 / W-1 exam unit",
    "GunPointMaleVersusFemale__impulse_v2": "M-1 margin-gate unit",
}

ARM_STATIC = s1.ARM_STATIC
ARM_A3 = s1.ARM_A3
ARM_K0 = s1.ARM_K0
ARM_A5 = s1.ARM_A5

# Two pre-frozen injection seeds; the second forward run differs only here.
SEEDS = {"r1": 20260827, "r2": 20260828}


class Stop(Exception):
    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__("%s: %s" % (verdict, reason))
        self.verdict = verdict
        self.reason = reason


# =========================================================================== #
# Part 0 -- half-protocol margins and mechanical course selection (0 LLM)
# =========================================================================== #
def _counts(slice_row: Mapping[str, Any]) -> tuple[int, int, int] | None:
    """Recover integer counts from a sealed slice.

    An empty quarter (ps0b records n=0 with null accuracies for a couple of
    tiny units) carries no counts to concatenate, and a role whose halves
    cannot both be read is not a half-protocol surface.
    """
    n = int(slice_row.get("n") or 0)
    identity = slice_row.get("identity_accuracy")
    program = slice_row.get("program_accuracy")
    if n <= 0 or identity is None or program is None:
        return None
    return (n, int(round(float(identity) * n)), int(round(float(program) * n)))


def _role_concat(slices: Mapping[str, Any], role: str) -> dict[str, Any] | None:
    """M-1's composition: concat the two same-role quarters, keep both gates.

    Not ps0b's stored ``half_slices``, which concatenates support with delayed
    inside a round and would collapse the dual gate into one surface.
    """
    parts = ["r1_%s" % role, "r2_%s" % role]
    if any(name not in slices for name in parts):
        return None
    n = identity = program = 0
    for name in parts:
        counted = _counts(slices[name])
        if counted is None:
            return None
        part_n, part_id, part_prog = counted
        n += part_n
        identity += part_id
        program += part_prog
    if n <= 0:
        return None
    reading = program / n - identity / n
    material = 1.0 / n
    return {"role": role, "n": n, "composed_of": parts,
            "identity_correct": identity, "program_correct": program,
            "reading": reading, "material_line": material,
            "margin_multiplier": reading / material if material else 0.0,
            "meets_material": reading >= material}


def half_protocol_margin(pair: Mapping[str, Any]) -> dict[str, Any] | None:
    slices = pair.get("slices") or {}
    support = _role_concat(slices, "support")
    delayed = _role_concat(slices, "delayed")
    if support is None or delayed is None:
        return None
    margin = min(support["margin_multiplier"], delayed["margin_multiplier"])
    return {
        "support": support, "delayed": delayed,
        "min_margin_multiplier": margin,
        "both_meet_2x": bool(support["margin_multiplier"] + 1e-12 >= MARGIN_BAR
                             and delayed["margin_multiplier"] + 1e-12
                             >= MARGIN_BAR),
        "quarter_margin_multiplier": pair.get("margin_multiplier"),
    }


def _oracle(unit_id: str) -> dict[str, Any] | None:
    path = ORACLE_DIR / ("%s.json" % unit_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pool() -> list[dict[str, Any]]:
    """Every sealed unit, annotated with the half-protocol arithmetic."""
    ps0b = json.loads(PS0B_JSON.read_text(encoding="utf-8"))
    by_unit: dict[str, list[Mapping[str, Any]]] = {}
    for pair in ps0b.get("pairs") or []:
        by_unit.setdefault(str(pair["unit_id"]), []).append(pair)

    rows: list[dict[str, Any]] = []
    for path in sorted(ORACLE_DIR.glob("*.json")):
        unit_id = path.stem
        oracle = _oracle(unit_id)
        if oracle is None:
            continue
        oracle_set = [str(op) for op in (oracle.get("oracle_set") or [])]
        hampel_pair = next(
            (pair for pair in by_unit.get(unit_id, ())
             if str(pair.get("program")) == TARGET_OPERATOR), None)
        half = half_protocol_margin(hampel_pair) if hampel_pair else None
        rows.append({
            "unit_id": unit_id,
            "dataset": oracle.get("dataset"),
            "injection": oracle.get("injection"),
            "series_length": oracle.get("series_length"),
            "name_family": (hampel_pair or {}).get("name_family"),
            "oracle_set": oracle_set,
            "menu_oracle_program": oracle.get("menu_oracle_program"),
            "menu_oracle_heldout_utility": oracle.get(
                "menu_oracle_heldout_utility"),
            "hampel_in_oracle_set": TARGET_OPERATOR in oracle_set,
            "census_learnability": (hampel_pair or {}).get(
                "census_learnability"),
            "quarter_grade": (hampel_pair or {}).get("grade"),
            "quarter_margin": (hampel_pair or {}).get("margin_multiplier"),
            "half_protocol": half,
            "half_margin": (half or {}).get("min_margin_multiplier"),
            "half_meets_2x": bool((half or {}).get("both_meet_2x")),
            "pattern": dict(oracle.get("public_features_binned") or {}),
            "n_slice_half_min": min(
                (half or {}).get("support", {}).get("n", 10 ** 9),
                (half or {}).get("delayed", {}).get("n", 10 ** 9))
            if half else None,
            "excluded_because": EXCLUDED.get(unit_id),
        })
    return rows


def _scope_of(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Five-axis intersection over candidate producer units."""
    supply_rows = [{
        "task_episode_id": row["unit_id"], "unit_id": row["unit_id"],
        "program": TARGET_OPERATOR, "relation": "POSITIVE",
        "conditioned_snapshot": False,
        "task_kind": "classification",
        "consumer_id": "ridge-raw-plus-difference-v1",
        "metric": "accuracy",
        "pattern": dict(row["pattern"]),
        "support_gain": 0.0, "delayed_gain": 0.0,
    } for row in rows]
    return ss.five_axis_scope(supply_rows)


def select_course() -> dict[str, Any]:
    pool = _pool()
    by_id = {row["unit_id"]: row for row in pool}

    # --- producers: hampel, half-margin >= 2x, not spent on another book ---
    producer_candidates = sorted(
        (row for row in pool
         if row["hampel_in_oracle_set"]
         and row["half_meets_2x"]
         and not row["excluded_because"]
         and row["census_learnability"] == "LEARNABLE"),
        key=lambda row: -float(row["half_margin"] or 0.0))
    producers = producer_candidates[:2]
    scope = _scope_of(producers) if len(producers) == 2 else None
    pattern = dict((scope or {}).get("pattern_intersection") or {})

    # --- beneficiary: machine-matches the producers' own Scope -------------
    ast = {"all": [{"feature": "task_kind", "op": "==",
                    "value": "classification"}]
           + [{"feature": key, "op": "==", "value": value}
              for key, value in sorted(pattern.items())]}
    beneficiary_scored = []
    for row in pool:
        if row["unit_id"] in {p["unit_id"] for p in producers}:
            continue
        if row["excluded_because"] or not row["hampel_in_oracle_set"]:
            continue
        features = {"task_kind": "classification", **row["pattern"]}
        matched, _score = evaluate_applicability(ast, features)
        beneficiary_scored.append({
            "unit_id": row["unit_id"], "machine_match": bool(matched),
            "half_margin": row["half_margin"],
            "half_meets_2x": row["half_meets_2x"],
            "census_learnability": row["census_learnability"]})
    # A beneficiary has to be able to *benefit*.  The book's fallback relaxes
    # the margin band, not held-in learnability: on a HELDOUT_ONLY unit the
    # Target's own feedback cannot approve the supplied family by
    # construction ("visible to the examiner, unlearnable by the student"),
    # so such a unit is a veto field, not a benefit field.  Accepting one
    # here would pre-decide NO_TRANSFER for a property of the pool rather
    # than of the Harness -- the S1a-r2 failure this precheck exists to catch.
    learnable = [row for row in beneficiary_scored
                 if row["machine_match"]
                 and row["census_learnability"] == "LEARNABLE"]
    strict = [row for row in learnable if row["half_meets_2x"]]
    strict.sort(key=lambda row: -float(row["half_margin"] or 0.0))
    learnable.sort(key=lambda row: -float(row["half_margin"] or 0.0))
    beneficiary = (strict[0] if strict else
                   (learnable[0] if learnable else None))
    beneficiary_note = (
        "strict: machine Scope match, held-in LEARNABLE, half-margin >= 2x"
        if strict else
        "fallback per book: machine Scope match and held-in LEARNABLE, "
        "highest half-margin; margin band recorded" if learnable else
        "no held-in LEARNABLE unit machine-matches the producers' Scope "
        "once the units spent on other books are excluded")

    # --- identity and HELDOUT_ONLY governance units ------------------------
    spent = {p["unit_id"] for p in producers}
    if beneficiary:
        spent.add(beneficiary["unit_id"])
    # "families as distinct as the pool allows": one unit per dataset across
    # the governance slots, so an identity reading and a HELDOUT_ONLY reading
    # are not two views of the same substrate.
    used_datasets = {by_id[unit]["dataset"] for unit in spent}

    def _take(candidates, count):
        taken = []
        for row in candidates:
            if len(taken) >= count:
                break
            if row["unit_id"] in spent or row["dataset"] in used_datasets:
                continue
            taken.append(row)
            spent.add(row["unit_id"])
            used_datasets.add(row["dataset"])
        return taken

    identity_pool = sorted(
        (row for row in pool
         if row["menu_oracle_program"] == "identity"
         and not row["excluded_because"]),
        key=lambda row: str(row["unit_id"]))
    heldout_pool = sorted(
        (row for row in pool
         if row["census_learnability"] == "HELDOUT_ONLY"
         and not row["excluded_because"]),
        key=lambda row: str(row["unit_id"]))
    identities = _take(identity_pool, 2)
    heldout = _take(heldout_pool, 1)

    course: list[dict[str, Any]] = []
    if len(producers) == 2 and beneficiary and len(identities) == 2 and heldout:
        order = [
            (producers[0]["unit_id"], "producer_A"),
            (identities[0]["unit_id"], "identity_A"),
            (producers[1]["unit_id"], "producer_B"),
            (beneficiary["unit_id"], "beneficiary"),
            (heldout[0]["unit_id"], "heldout_only"),
            (identities[1]["unit_id"], "identity_B"),
        ]
        for position, (unit_id, role) in enumerate(order, start=1):
            row = by_id[unit_id]
            course.append({
                "position": position, "unit_id": unit_id, "role": role,
                "dataset": row["dataset"], "injection": row["injection"],
                "series_length": row["series_length"],
                "menu_oracle_program": row["menu_oracle_program"],
                "menu_oracle_heldout_utility": row[
                    "menu_oracle_heldout_utility"],
                "half_margin": row["half_margin"],
                "half_meets_2x": row["half_meets_2x"],
                "census_learnability": row["census_learnability"],
                "n_slice_half_min": row["n_slice_half_min"],
                "name_family": row["name_family"],
            })

    # --- treatment-group existence precheck (arithmetic, before any LLM) ---
    boundary = 3 if course else None
    precheck = {
        "producers_found": len(producers),
        "producer_ids": [row["unit_id"] for row in producers],
        "producers_distinct_task_episode_id": (
            len({row["unit_id"] for row in producers}) == 2),
        "five_axis_scope_non_empty": bool(pattern),
        "pattern_intersection_leaves": sorted(pattern),
        "beneficiary": beneficiary,
        "beneficiary_rule": beneficiary_note,
        "identity_units": [row["unit_id"] for row in identities],
        "heldout_only_unit": [row["unit_id"] for row in heldout],
        "expected_card_boundary_after_position": boundary,
        "expected_first_divergence_position": (
            boundary + 1 if boundary else None),
        "supply_tier_rule": (
            "P0 compile_supply_tier: 2 distinct unguided POSITIVE Episodes "
            "of one Program family, five-axis Scope non-empty, no opposing "
            "reading -- supplies_candidates only"),
    }
    constructible = bool(
        course
        and precheck["producers_distinct_task_episode_id"]
        and precheck["five_axis_scope_non_empty"]
        and beneficiary is not None)

    # What would unblock a course, stated so the main line can decide rather
    # than guess.  Each row is an excluded unit re-scored as if it were free.
    counterfactual = []
    for unit_id, spent_on in sorted(EXCLUDED.items()):
        row = by_id.get(unit_id)
        if row is None or unit_id in {p["unit_id"] for p in producers}:
            continue
        features = {"task_kind": "classification", **row["pattern"]}
        matched, _score = evaluate_applicability(ast, features)
        counterfactual.append({
            "unit_id": unit_id,
            "spent_on": spent_on,
            "machine_match_producer_scope": bool(matched),
            "census_learnability": row["census_learnability"],
            "half_margin": row["half_margin"],
            "would_qualify_as_beneficiary": bool(
                matched and row["census_learnability"] == "LEARNABLE"),
            "cost_of_releasing": (
                "re-using a dual-source unit would make the card partly "
                "'brought in' rather than earned inside the course, which is "
                "the distinction this exam exists to draw"
                if "dual-source" in spent_on else
                "re-using an exam unit weakens independence from the book "
                "that already read it"),
        })
    # Delta_material for the regret gate: the coarsest half-protocol slice
    # across the course.
    deltas = [row["n_slice_half_min"] for row in course
              if row["n_slice_half_min"]]
    delta_material = max((1.0 / n for n in deltas), default=None)
    return {
        "pool": pool,
        "producer_candidates": [
            {"unit_id": row["unit_id"], "half_margin": row["half_margin"],
             "quarter_margin": row["quarter_margin"],
             "half_meets_2x": row["half_meets_2x"]}
            for row in producer_candidates],
        "excluded": EXCLUDED,
        "producers_scope_v1": scope,
        "beneficiary_scored": beneficiary_scored,
        "course": course,
        "transfer_graph": (
            [{"from": [precheck["producer_ids"][0],
                       precheck["producer_ids"][1]],
              "via": "Slow boundary after position %s -> compile_supply_tier"
                     % boundary,
              "to": beneficiary["unit_id"] if beneficiary else None,
              "carrier": "supplies_candidates card (grants_execution=false)"}]
            if constructible else []),
        "precheck": precheck,
        "counterfactual_if_an_exclusion_were_released": counterfactual,
        "first_unmet_condition": (
            None if constructible
            else "no beneficiary: every hampel-bearing unit left after the "
                 "four book exclusions is HELDOUT_ONLY, so held-in feedback "
                 "cannot approve the supplied family on any of them"),
        "constructible": constructible,
        "delta_material": delta_material,
        "seeds": dict(SEEDS),
        "protocol": {
            "slicing": "M-1 half protocol (one held-in round; Support = "
                       "concat(r1_support, r2_support), delayed = "
                       "concat(r1_delayed, r2_delayed))",
            "arms": [ARM_STATIC, ARM_A3, ARM_K0, ARM_A5],
            "k0": "bootstrap three cards + the inert Slow card; no "
                  "Target-local capability and no PS dual-source card",
            "llm_per_unit_per_arm": 12,
            "fit_per_unit_per_arm": 10,
            "llm_per_slow_boundary": 6,
        },
    }


def _freeze_markdown(payload: Mapping[str, Any]) -> str:
    pre = payload["precheck"]
    lines = [
        "# S1-v2 course freeze (Part 0, 0 LLM)",
        "",
        "protocol: `%s`  git: `%s`" % (payload["protocol_version"],
                                       payload["git_head"]),
        "", "**%s**" % payload["verdict"]["verdict"], "",
        payload["verdict"]["reason"], "",
        "## Half-protocol margins (recomputed from ps0b sealed counts)", "",
        "| unit | hampel in oracle | census | quarter margin | **half "
        "margin** | half >= 2x | excluded |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(payload["pool"],
                      key=lambda r: -(r["half_margin"] or -1)):
        if row["half_margin"] is None and not row["hampel_in_oracle_set"]:
            continue
        lines.append("| `%s` | %s | %s | %s | **%s** | %s | %s |" % (
            row["unit_id"], row["hampel_in_oracle_set"],
            row["census_learnability"],
            ("%.2f" % row["quarter_margin"]) if row["quarter_margin"]
            else "-",
            ("%.2f" % row["half_margin"]) if row["half_margin"] else "-",
            row["half_meets_2x"], row["excluded_because"] or ""))
    lines += ["", "## Course (frozen order)", "",
              "| # | role | unit | menu oracle | half margin | census | "
              "coarsest half n |", "|---|---|---|---|---|---|---|"]
    for row in payload["course"]:
        lines.append("| %d | %s | `%s` | `%s` | %s | %s | %s |" % (
            row["position"], row["role"], row["unit_id"],
            row["menu_oracle_program"],
            ("%.2f" % row["half_margin"]) if row["half_margin"] else "-",
            row["census_learnability"], row["n_slice_half_min"]))
    lines += ["", "## Transfer graph", ""]
    for edge in payload["transfer_graph"]:
        lines.append("- `%s` + `%s` --%s--> `%s` (carrier: %s)" % (
            edge["from"][0], edge["from"][1], edge["via"], edge["to"],
            edge["carrier"]))
    if not payload["transfer_graph"]:
        lines.append("- (none: the course is not constructible)")
    lines += ["", "## Treatment-group precheck (arithmetic, pre-LLM)", "",
              "- producers: %s" % ", ".join(pre["producer_ids"]),
              "- distinct task_episode_id: %s"
              % pre["producers_distinct_task_episode_id"],
              "- five-axis Scope non-empty: %s (%d pattern leaves)"
              % (pre["five_axis_scope_non_empty"],
                 len(pre["pattern_intersection_leaves"])),
              "- beneficiary: %s -- %s"
              % ((pre["beneficiary"] or {}).get("unit_id"),
                 pre["beneficiary_rule"]),
              "- identity units: %s" % ", ".join(pre["identity_units"]),
              "- HELDOUT_ONLY unit: %s" % ", ".join(pre["heldout_only_unit"]),
              "- expected card boundary: after position %s"
              % pre["expected_card_boundary_after_position"],
              "- expected first divergence: position %s"
              % pre["expected_first_divergence_position"],
              "",
              "- **Delta_material** (regret gate) = max_u(1/n_slice_u) = %s"
              % payload["delta_material"], ""]
    if payload.get("first_unmet_condition"):
        lines += ["## First unmet condition", "",
                  "- %s" % payload["first_unmet_condition"], "",
                  "### Candidate beneficiaries, scored", "",
                  "| unit | Scope match | held-in census | half margin |",
                  "|---|---|---|---|"]
        for row in payload["beneficiary_scored"]:
            lines.append("| `%s` | %s | %s | %s |" % (
                row["unit_id"], row["machine_match"],
                row["census_learnability"],
                ("%.2f" % row["half_margin"]) if row["half_margin"] else "-"))
        lines += ["", "### If an exclusion were released", "",
                  "| unit | spent on | Scope match | census | half margin | "
                  "would qualify |", "|---|---|---|---|---|---|"]
        for row in payload["counterfactual_if_an_exclusion_were_released"]:
            lines.append("| `%s` | %s | %s | %s | %s | **%s** |" % (
                row["unit_id"], row["spent_on"],
                row["machine_match_producer_scope"],
                row["census_learnability"],
                ("%.2f" % row["half_margin"]) if row["half_margin"] else "-",
                row["would_qualify_as_beneficiary"]))
        lines.append("")
    return "\n".join(lines) + "\n"


def freeze() -> int:
    selection = select_course()
    constructible = selection["constructible"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "design": "docs/S1V2_DESIGN_DRAFT_2026-08-27.md",
        "sources": {
            "margins": PS0B_JSON.relative_to(PROJECT_ROOT).as_posix(),
            "oracles": "artifacts/functional/e2/s1_oracle/*.json "
                       "(exam keys only; not loaded into any arm)",
        },
        **selection,
        "verdict": {
            "verdict": ("S1V2_COURSE_FROZEN" if constructible
                        else "COURSE_NOT_CONSTRUCTIBLE"),
            "reason": (
                "two producers with distinct task_episode_id, a non-empty "
                "five-axis Scope and a matching beneficiary after them: the "
                "treatment group exists on arithmetic before any LLM is "
                "spent." if constructible else
                "the arithmetic precheck did not produce a course; no live "
                "run is started.  See the precheck block for the first "
                "unmet condition."),
        },
        "ledger": {"llm": 0, "consumer_fits": 0, "downloads": 0},
    }
    FREEZE_JSON.write_text(
        json.dumps(s1._plain(payload), ensure_ascii=False, indent=1,
                   sort_keys=True, default=str) + "\n", encoding="utf-8")
    FREEZE_MD.write_text(_freeze_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"]["verdict"],
        "producers": selection["precheck"]["producer_ids"],
        "beneficiary": (selection["precheck"]["beneficiary"] or {}).get(
            "unit_id"),
        "course": [row["unit_id"] for row in selection["course"]],
        "delta_material": selection["delta_material"],
        "artifact": str(FREEZE_JSON),
    }, ensure_ascii=False, indent=1))
    return 0 if constructible else 1


# =========================================================================== #
# Part 0 r2 -- re-freeze under the main line's two rulings
# =========================================================================== #
# Ruling (a): the two exam units are released as beneficiaries, strong and
# weak, with the margin stratification pre-registered.  The dual-source pair
# stays excluded, because a card compiled partly from a unit the course also
# examines is no longer "earned inside the course".
EXCLUDED_R2 = {
    "GunPointAgeSpan__impulse_v2": "dual-source A (PS-0)",
    "PowerCons__impulse_v2": "dual-source B (PS-0c)",
}
PRODUCER_A = "PowerCons__burst_cls2"
PRODUCER_B = "GunPoint__impulse_v2"
BENEFICIARY_STRONG = "GunPointOldVersusYoung__impulse_v2"
BENEFICIARY_WEAK = "GunPointMaleVersusFemale__impulse_v2"

PRIOR_EXPOSURE_NOTE = (
    "Both beneficiaries are units whose hampel convertibility is already "
    "known from PS-2 / W-1 (GPOvY) and M-1 (GPMvF).  There is no leakage "
    "into any arm: A5-online starts from K0 and its only Source knowledge is "
    "what this course compiles from its own Episodes.  The novel claim is "
    "therefore the end-to-end ITT compounding -- knowledge produced inside "
    "the course changing later units -- and explicitly NOT that these units "
    "are convertible, which is prior work."
)


def _material_line(row: Mapping[str, Any]) -> float | None:
    n = row.get("n_slice_half_min")
    return (1.0 / float(n)) if n else None


def select_course_r2() -> dict[str, Any]:
    pool = _pool()
    by_id = {row["unit_id"]: row for row in pool}
    missing = [unit for unit in (PRODUCER_A, PRODUCER_B, BENEFICIARY_STRONG,
                                 BENEFICIARY_WEAK) if unit not in by_id]
    if missing:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "named units absent from the sealed roster: %s"
                   % ", ".join(missing))

    producers = [by_id[PRODUCER_A], by_id[PRODUCER_B]]
    scope = _scope_of(producers)
    pattern = dict((scope or {}).get("pattern_intersection") or {})
    ast = {"all": [{"feature": "task_kind", "op": "==",
                    "value": "classification"}]
           + [{"feature": key, "op": "==", "value": value}
              for key, value in sorted(pattern.items())]}

    beneficiaries = []
    for unit_id, band in ((BENEFICIARY_STRONG, "strong"),
                          (BENEFICIARY_WEAK, "weak")):
        row = by_id[unit_id]
        features = {"task_kind": "classification", **row["pattern"]}
        matched, _score = evaluate_applicability(ast, features)
        beneficiaries.append({
            "unit_id": unit_id, "margin_band": band,
            "machine_match_producer_scope": bool(matched),
            "census_learnability": row["census_learnability"],
            "half_margin": row["half_margin"],
            "half_meets_2x": row["half_meets_2x"],
            "material_line": _material_line(row),
            "n_slice_half_min": row["n_slice_half_min"],
            "prior_exposure": ("PS-2 / W-1 exam unit"
                               if unit_id == BENEFICIARY_STRONG
                               else "M-1 margin-gate unit"),
        })

    spent = {PRODUCER_A, PRODUCER_B, BENEFICIARY_STRONG, BENEFICIARY_WEAK}
    used_datasets = {by_id[unit]["dataset"] for unit in spent}

    def _take(candidates, count):
        taken = []
        for row in candidates:
            if len(taken) >= count:
                break
            if row["unit_id"] in spent or row["dataset"] in used_datasets:
                continue
            taken.append(row)
            spent.add(row["unit_id"])
            used_datasets.add(row["dataset"])
        return taken

    identities = _take(sorted(
        (row for row in pool
         if row["menu_oracle_program"] == "identity"
         and row["unit_id"] not in EXCLUDED_R2),
        key=lambda row: str(row["unit_id"])), 2)
    heldout = _take(sorted(
        (row for row in pool
         if row["census_learnability"] == "HELDOUT_ONLY"
         and row["unit_id"] not in EXCLUDED_R2),
        key=lambda row: str(row["unit_id"])), 1)

    order = [
        (PRODUCER_A, "producer_A"),
        (identities[0]["unit_id"], "identity_A"),
        (PRODUCER_B, "producer_B"),
        (BENEFICIARY_STRONG, "beneficiary_strong"),
        (BENEFICIARY_WEAK, "beneficiary_weak"),
        (heldout[0]["unit_id"], "heldout_only"),
        (identities[1]["unit_id"], "identity_B"),
    ]
    course = []
    for position, (unit_id, role) in enumerate(order, start=1):
        row = by_id[unit_id]
        course.append({
            "position": position, "unit_id": unit_id, "role": role,
            "dataset": row["dataset"], "injection": row["injection"],
            "series_length": row["series_length"],
            "menu_oracle_program": row["menu_oracle_program"],
            "menu_oracle_heldout_utility": row["menu_oracle_heldout_utility"],
            "half_margin": row["half_margin"],
            "half_meets_2x": row["half_meets_2x"],
            "census_learnability": row["census_learnability"],
            "n_slice_half_min": row["n_slice_half_min"],
            "name_family": row["name_family"],
        })

    # Ruling (b): the regret gate is the sum of the two beneficiaries' own
    # half-protocol material lines, not the coarsest slice in the course.
    delta_parts = [row["material_line"] for row in beneficiaries]
    delta_material = (sum(part for part in delta_parts if part)
                      if all(delta_parts) else None)
    constructible = bool(
        all(row["machine_match_producer_scope"] for row in beneficiaries)
        and all(row["census_learnability"] == "LEARNABLE"
                for row in beneficiaries)
        and pattern and len(identities) == 2 and heldout)
    return {
        "pool": pool,
        "excluded": EXCLUDED_R2,
        "rulings": {
            "a_beneficiaries_released": {
                "released": [BENEFICIARY_STRONG, BENEFICIARY_WEAK],
                "still_excluded": sorted(EXCLUDED_R2),
                "stratified_prediction": (
                    "pre-registered: A5's advantage should concentrate on the "
                    "strong-margin beneficiary and be marginal on the weak "
                    "one"),
                "prior_exposure_note": PRIOR_EXPOSURE_NOTE,
            },
            "b_regret_gate": {
                "definition": "sum of the two beneficiaries' half-protocol "
                              "material lines",
                "parts": {row["unit_id"]: row["material_line"]
                          for row in beneficiaries},
                "delta_material": delta_material,
                "cost_gate_unchanged": "convertible units average >= 1 probe "
                                       "saved",
            },
        },
        "producers_scope_v1": scope,
        "beneficiaries": beneficiaries,
        "course": course,
        "course_length": len(course),
        "course_length_note": (
            "seven units; the book's 'eight' counts the Slow boundary between "
            "producer B and the strong beneficiary as a step"),
        "transfer_graph": [{
            "from": [PRODUCER_A, PRODUCER_B],
            "via": "Slow boundary after position 3 -> compile_supply_tier",
            "to": [BENEFICIARY_STRONG, BENEFICIARY_WEAK],
            "carrier": "supplies_candidates card (grants_execution=false)",
        }],
        "precheck": {
            "expected_card_boundary_after_position": 3,
            "expected_first_divergence_position": 4,
            "expected_first_divergence_unit": BENEFICIARY_STRONG,
            "five_axis_scope_non_empty": bool(pattern),
            "pattern_intersection_leaves": sorted(pattern),
            "beneficiaries_machine_match": all(
                row["machine_match_producer_scope"] for row in beneficiaries),
        },
        "constructible": constructible,
        "delta_material": delta_material,
        "seeds": dict(SEEDS),
        "protocol": {
            "slicing": "M-1 half protocol (one held-in round; Support = "
                       "concat(r1_support, r2_support), delayed = "
                       "concat(r1_delayed, r2_delayed))",
            "arms": [ARM_STATIC, ARM_A3, ARM_K0, ARM_A5],
            "k0": "bootstrap three cards + the inert Slow card; no "
                  "Target-local capability and no PS dual-source card",
            "llm_per_unit_per_arm": 12,
            "fit_per_unit_per_arm": 10,
            "llm_per_slow_boundary": 6,
        },
    }


def _freeze_r2_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# S1-v2 course freeze r2 (Part 0 after the main line's rulings)",
        "",
        "protocol: `%s`  git: `%s`" % (payload["protocol_version"],
                                       payload["git_head"]),
        "", "**%s**" % payload["verdict"]["verdict"], "",
        payload["verdict"]["reason"], "",
        "> %s" % payload["rulings"]["a_beneficiaries_released"][
            "prior_exposure_note"],
        "", "## Course (frozen forward order)", "",
        "| # | role | unit | menu oracle | half margin | census | coarsest "
        "half n |", "|---|---|---|---|---|---|---|",
    ]
    for row in payload["course"]:
        lines.append("| %d | %s | `%s` | `%s` | %s | %s | %s |" % (
            row["position"], row["role"], row["unit_id"],
            row["menu_oracle_program"],
            ("%.2f" % row["half_margin"]) if row["half_margin"] else "-",
            row["census_learnability"], row["n_slice_half_min"]))
    lines += ["", "- %s" % payload["course_length_note"], "",
              "## Beneficiaries (ruling a)", "",
              "| unit | band | Scope match | census | half margin | material "
              "line | prior exposure |", "|---|---|---|---|---|---|---|"]
    for row in payload["beneficiaries"]:
        lines.append("| `%s` | **%s** | %s | %s | %.2f | %.4f | %s |" % (
            row["unit_id"], row["margin_band"],
            row["machine_match_producer_scope"], row["census_learnability"],
            row["half_margin"], row["material_line"], row["prior_exposure"]))
    gate = payload["rulings"]["b_regret_gate"]
    lines += ["", "- stratified prediction: %s"
              % payload["rulings"]["a_beneficiaries_released"][
                  "stratified_prediction"], "",
              "## Gates (ruling b)", "",
              "- regret gate `Delta_material` = %s = %.6f"
              % (" + ".join("%.6f" % v for v in gate["parts"].values()),
                 gate["delta_material"]),
              "- cost gate: %s" % gate["cost_gate_unchanged"], "",
              "## Transfer graph", ""]
    for edge in payload["transfer_graph"]:
        lines.append("- `%s` + `%s` --%s--> %s (carrier: %s)" % (
            edge["from"][0], edge["from"][1], edge["via"],
            ", ".join("`%s`" % unit for unit in edge["to"]), edge["carrier"]))
    pre = payload["precheck"]
    lines += ["", "## Precheck", "",
              "- five-axis Scope non-empty: %s (%d leaves)"
              % (pre["five_axis_scope_non_empty"],
                 len(pre["pattern_intersection_leaves"])),
              "- both beneficiaries machine-match: %s"
              % pre["beneficiaries_machine_match"],
              "- expected card boundary: after position %s"
              % pre["expected_card_boundary_after_position"],
              "- expected first divergence: position %s (`%s`)"
              % (pre["expected_first_divergence_position"],
                 pre["expected_first_divergence_unit"]),
              "- seeds: %s" % json.dumps(payload["seeds"]), ""]
    return "\n".join(lines) + "\n"


def freeze_r2() -> int:
    selection = select_course_r2()
    constructible = selection["constructible"]
    payload = {
        "protocol_version": PROTOCOL_VERSION + "_r2",
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "design": "docs/S1V2_DESIGN_DRAFT_2026-08-27.md",
        "supersedes": FREEZE_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "sources": {
            "margins": PS0B_JSON.relative_to(PROJECT_ROOT).as_posix(),
            "oracles": "artifacts/functional/e2/s1_oracle/*.json "
                       "(exam keys only; not loaded into any arm)",
        },
        **selection,
        "verdict": {
            "verdict": ("S1V2_COURSE_FROZEN_R2" if constructible
                        else "COURSE_NOT_CONSTRUCTIBLE"),
            "reason": (
                "two producers with distinct task_episode_id and a non-empty "
                "five-axis Scope, followed by two held-in learnable "
                "beneficiaries that machine-match that Scope at separated "
                "margin bands.  The treatment group exists on arithmetic."
                if constructible else
                "the named units did not satisfy the precheck"),
        },
        "ledger": {"llm": 0, "consumer_fits": 0, "downloads": 0},
    }
    FREEZE_R2_JSON.write_text(
        json.dumps(s1._plain(payload), ensure_ascii=False, indent=1,
                   sort_keys=True, default=str) + "\n", encoding="utf-8")
    FREEZE_R2_MD.write_text(_freeze_r2_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"]["verdict"],
        "course": [row["unit_id"] for row in selection["course"]],
        "delta_material": selection["delta_material"],
        "artifact": str(FREEZE_R2_JSON),
    }, ensure_ascii=False, indent=1))
    return 0 if constructible else 1


# =========================================================================== #
# Part 0 v3 -- arbitration A: producers picked on cold-discovery rate
# =========================================================================== #
COURSE_NAME_V3 = "discovery-reliable development curriculum"
PRODUCER_A_V3 = "GunPointAgeSpan__impulse_v2"
PRODUCER_B_V3 = "PowerCons__impulse_v2"
# Demonstrated cold-discovery rate: how often an arm with no card naming the
# family proposed and earned it on this unit, across the line's own books.
COLD_DISCOVERY = {
    PRODUCER_A_V3: {"earned": 2, "attempts": 2, "source": "PS-0 re-earn"},
    PRODUCER_B_V3: {"earned": 2, "attempts": 3, "source": "PS-0c re-earn"},
}
EXCLUSION_SEMANTICS_V3 = (
    "Revised, and the revision is what releases the former dual-source pair "
    "as producers.  The constraint that protects 'the card is earned inside "
    "the course' is not 'this unit was ever a source elsewhere'; it is (i) K0 "
    "carries no card, so A5 starts with nothing, and (ii) no beneficiary is "
    "also a producer, so nothing is graded on the unit that taught it.  "
    "Re-earning the family on a producer *inside* this course is exactly what "
    "the course is supposed to do, and it does not make the compiled card "
    "'brought in' -- the Episodes it compiles from are this course's own.  "
    "Checked with sol."
)
FAMILY_OVERLAP_NOTE_V3 = (
    "GunPointAgeSpan (producer A) and GunPointOldVersusYoung / "
    "GunPointMaleVersusFemale (beneficiaries) share the GunPoint name family. "
    "The units themselves are disjoint and no beneficiary is a producer, but "
    "this is a within-family transfer at the substrate level and must not be "
    "reported as cross-family capability."
)
NATURAL_BOOTSTRAP_CONTROL = (
    "Course r1 (natural-bootstrap producers, chosen on sealed margin alone) "
    "returned TREATMENT_EMPTY: the arm never proposed the family on either "
    "producer, so no card compiled.  That run is retained as the discovery "
    "module's control -- it is the measurement of what happens when producer "
    "selection ignores proposability.  Artifact: "
    "artifacts/functional/e2/s1v2_forward_run1.json"
)


def select_course_v3() -> dict[str, Any]:
    pool = _pool()
    by_id = {row["unit_id"]: row for row in pool}
    named = [PRODUCER_A_V3, PRODUCER_B_V3, BENEFICIARY_STRONG,
             BENEFICIARY_WEAK]
    missing = [unit for unit in named if unit not in by_id]
    if missing:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "named units absent: %s" % ", ".join(missing))

    producers = [by_id[PRODUCER_A_V3], by_id[PRODUCER_B_V3]]
    scope = _scope_of(producers)
    pattern = dict((scope or {}).get("pattern_intersection") or {})
    ast = {"all": [{"feature": "task_kind", "op": "==",
                    "value": "classification"}]
           + [{"feature": key, "op": "==", "value": value}
              for key, value in sorted(pattern.items())]}

    beneficiaries = []
    for unit_id, band in ((BENEFICIARY_STRONG, "strong"),
                          (BENEFICIARY_WEAK, "weak")):
        row = by_id[unit_id]
        matched, _score = evaluate_applicability(
            ast, {"task_kind": "classification", **row["pattern"]})
        beneficiaries.append({
            "unit_id": unit_id, "margin_band": band,
            "machine_match_producer_scope": bool(matched),
            "census_learnability": row["census_learnability"],
            "half_margin": row["half_margin"],
            "half_meets_2x": row["half_meets_2x"],
            "material_line": _material_line(row),
            "n_slice_half_min": row["n_slice_half_min"],
            "prior_exposure": ("PS-2 / W-1 exam unit"
                               if unit_id == BENEFICIARY_STRONG
                               else "M-1 margin-gate unit"),
            "is_also_a_producer": unit_id in (PRODUCER_A_V3, PRODUCER_B_V3),
        })

    order = [
        (PRODUCER_A_V3, "producer_A"),
        ("BeetleFly__impulse_v2", "identity_A"),
        (PRODUCER_B_V3, "producer_B"),
        (BENEFICIARY_STRONG, "beneficiary_strong"),
        (BENEFICIARY_WEAK, "beneficiary_weak"),
        ("Herring__impulse_v2", "heldout_only"),
        ("BirdChicken__burst_cls2", "identity_B"),
    ]
    course = []
    for position, (unit_id, role) in enumerate(order, start=1):
        row = by_id[unit_id]
        course.append({
            "position": position, "unit_id": unit_id, "role": role,
            "dataset": row["dataset"], "injection": row["injection"],
            "series_length": row["series_length"],
            "menu_oracle_program": row["menu_oracle_program"],
            "menu_oracle_heldout_utility": row["menu_oracle_heldout_utility"],
            "half_margin": row["half_margin"],
            "half_meets_2x": row["half_meets_2x"],
            "census_learnability": row["census_learnability"],
            "n_slice_half_min": row["n_slice_half_min"],
            "name_family": row["name_family"],
            "cold_discovery": COLD_DISCOVERY.get(unit_id),
        })

    delta_parts = [row["material_line"] for row in beneficiaries]
    delta_material = (sum(part for part in delta_parts if part)
                      if all(delta_parts) else None)
    constructible = bool(
        all(row["machine_match_producer_scope"] for row in beneficiaries)
        and all(row["census_learnability"] == "LEARNABLE"
                for row in beneficiaries)
        and not any(row["is_also_a_producer"] for row in beneficiaries)
        and pattern)
    return {
        "course_name": COURSE_NAME_V3,
        "course_name_basis": (
            "producers are selected on demonstrated cold-discovery rate; the "
            "natural-bootstrap course r1 is retained as the discovery "
            "module's control"),
        "pool": pool,
        "producer_selection": {
            "rule": "demonstrated cold-discovery rate on this unit, then "
                    "sealed half-protocol margin",
            "producers": [
                {"unit_id": row["unit_id"],
                 "cold_discovery": COLD_DISCOVERY[row["unit_id"]],
                 "half_margin": row["half_margin"],
                 "half_meets_2x": row["half_meets_2x"]}
                for row in producers],
            "why_r1_failed": (
                "r1 picked producers on sealed margin alone; a margin says a "
                "reading would be legible if the family were probed, not that "
                "the arm will propose it"),
        },
        "exclusion_semantics_revision": EXCLUSION_SEMANTICS_V3,
        "family_overlap_note": FAMILY_OVERLAP_NOTE_V3,
        "natural_bootstrap_control": NATURAL_BOOTSTRAP_CONTROL,
        "rulings": {
            "a_beneficiaries_released": {
                "released": [BENEFICIARY_STRONG, BENEFICIARY_WEAK],
                "stratified_prediction": (
                    "pre-registered: A5's advantage should concentrate on the "
                    "strong-margin beneficiary and be marginal on the weak "
                    "one"),
                "prior_exposure_note": PRIOR_EXPOSURE_NOTE,
            },
            "b_regret_gate": {
                "definition": "sum of the two beneficiaries' half-protocol "
                              "material lines",
                "parts": {row["unit_id"]: row["material_line"]
                          for row in beneficiaries},
                "delta_material": delta_material,
                "cost_gate_unchanged": "convertible units average >= 1 probe "
                                       "saved",
            },
        },
        "producers_scope_v1": scope,
        "beneficiaries": beneficiaries,
        "course": course,
        "course_length": len(course),
        "transfer_graph": [{
            "from": [PRODUCER_A_V3, PRODUCER_B_V3],
            "via": "Slow boundary after position 3 -> compile_supply_tier",
            "to": [BENEFICIARY_STRONG, BENEFICIARY_WEAK],
            "carrier": "supplies_candidates card (grants_execution=false)",
        }],
        "precheck": {
            "expected_card_boundary_after_position": 3,
            "expected_first_divergence_position": 4,
            "expected_first_divergence_unit": BENEFICIARY_STRONG,
            "five_axis_scope_non_empty": bool(pattern),
            "pattern_intersection_leaves": sorted(pattern),
            "beneficiaries_machine_match": all(
                row["machine_match_producer_scope"] for row in beneficiaries),
            "no_beneficiary_is_a_producer": not any(
                row["is_also_a_producer"] for row in beneficiaries),
            "k0_carries_no_card": "asserted at run time by compile_k0 purity",
        },
        "constructible": constructible,
        "delta_material": delta_material,
        "seeds": dict(SEEDS),
        "replicate_kind": "sampling",
        "replicate_semantics": (
            "the injection has no RNG to seed "
            "(run_e2_t6_cls_op_shared_harness.py:3896-3901), so a second run "
            "is a sampling replicate: identical substrate and protocol, Fast "
            "Agent the only stochastic element"),
        "protocol": {
            "slicing": "M-1 half protocol (one held-in round; Support = "
                       "concat(r1_support, r2_support), delayed = "
                       "concat(r1_delayed, r2_delayed))",
            "arms": [ARM_STATIC, ARM_A3, ARM_K0, ARM_A5],
            "k0": "bootstrap three cards + the inert Slow card; no "
                  "Target-local capability and no PS dual-source card",
            "llm_per_unit_per_arm": 12,
            "fit_per_unit_per_arm": 10,
            "llm_per_slow_boundary": 6,
        },
    }


def _freeze_v3_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# S1-v2 course freeze v3 -- %s" % payload["course_name"], "",
        "protocol: `%s`  git: `%s`" % (payload["protocol_version"],
                                       payload["git_head"]),
        "", "**%s**" % payload["verdict"]["verdict"], "",
        payload["verdict"]["reason"], "",
        "- %s" % payload["course_name_basis"], "",
        "> **Exclusion semantics, revised.** %s"
        % payload["exclusion_semantics_revision"], "",
        "> **Prior exposure.** %s"
        % payload["rulings"]["a_beneficiaries_released"][
            "prior_exposure_note"], "",
        "> **Family overlap.** %s" % payload["family_overlap_note"], "",
        "> **Control.** %s" % payload["natural_bootstrap_control"], "",
        "> **Replicates.** %s" % payload["replicate_semantics"], "",
        "## Course (frozen forward order)", "",
        "| # | role | unit | menu oracle | half margin | cold discovery | "
        "census |", "|---|---|---|---|---|---|---|",
    ]
    for row in payload["course"]:
        cold = row.get("cold_discovery")
        lines.append("| %d | %s | `%s` | `%s` | %s | %s | %s |" % (
            row["position"], row["role"], row["unit_id"],
            row["menu_oracle_program"],
            ("%.2f" % row["half_margin"]) if row["half_margin"] else "-",
            ("%d/%d (%s)" % (cold["earned"], cold["attempts"], cold["source"]))
            if cold else "-",
            row["census_learnability"]))
    lines += ["", "## Producer selection (the only change from r2)", "",
              "- rule: %s" % payload["producer_selection"]["rule"],
              "- why r1 failed: %s"
              % payload["producer_selection"]["why_r1_failed"], "",
              "## Beneficiaries", "",
              "| unit | band | Scope match | census | half margin | material "
              "line | also a producer |", "|---|---|---|---|---|---|---|"]
    for row in payload["beneficiaries"]:
        lines.append("| `%s` | **%s** | %s | %s | %.2f | %.4f | %s |" % (
            row["unit_id"], row["margin_band"],
            row["machine_match_producer_scope"], row["census_learnability"],
            row["half_margin"], row["material_line"],
            row["is_also_a_producer"]))
    gate = payload["rulings"]["b_regret_gate"]
    lines += ["", "- stratified prediction: %s"
              % payload["rulings"]["a_beneficiaries_released"][
                  "stratified_prediction"], "",
              "## Gates", "",
              "- regret gate `Delta_material` = %s = %.6f"
              % (" + ".join("%.6f" % v for v in gate["parts"].values()),
                 gate["delta_material"]),
              "- cost gate: %s" % gate["cost_gate_unchanged"], "",
              "## Transfer graph", ""]
    for edge in payload["transfer_graph"]:
        lines.append("- `%s` + `%s` --%s--> %s (carrier: %s)" % (
            edge["from"][0], edge["from"][1], edge["via"],
            ", ".join("`%s`" % unit for unit in edge["to"]), edge["carrier"]))
    pre = payload["precheck"]
    lines += ["", "## Precheck", "",
              "- five-axis Scope non-empty: %s (%d leaves)"
              % (pre["five_axis_scope_non_empty"],
                 len(pre["pattern_intersection_leaves"])),
              "- both beneficiaries machine-match: %s"
              % pre["beneficiaries_machine_match"],
              "- no beneficiary is a producer: %s"
              % pre["no_beneficiary_is_a_producer"],
              "- K0 purity: %s" % pre["k0_carries_no_card"],
              "- expected card boundary: after position %s"
              % pre["expected_card_boundary_after_position"],
              "- expected first divergence: position %s (`%s`)"
              % (pre["expected_first_divergence_position"],
                 pre["expected_first_divergence_unit"]), ""]
    return "\n".join(lines) + "\n"


def freeze_v3() -> int:
    selection = select_course_v3()
    constructible = selection["constructible"]
    payload = {
        "protocol_version": PROTOCOL_VERSION + "_v3",
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "design": "docs/S1V2_DESIGN_DRAFT_2026-08-27.md",
        "supersedes": FREEZE_R2_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "arbitration": "A (producer side only; scoring, ITT, gates, budgets, "
                       "half protocol, K0 purity and oracle discipline all "
                       "carry over from r2)",
        "sources": {
            "margins": PS0B_JSON.relative_to(PROJECT_ROOT).as_posix(),
            "oracles": "artifacts/functional/e2/s1_oracle/*.json "
                       "(exam keys only; not loaded into any arm)",
        },
        **selection,
        "verdict": {
            "verdict": ("S1V2_COURSE_FROZEN_V3" if constructible
                        else "COURSE_NOT_CONSTRUCTIBLE"),
            "reason": (
                "two producers with demonstrated cold-discovery on their own "
                "unit, a non-empty five-axis Scope, and two held-in learnable "
                "beneficiaries at separated margin bands, none of which is a "
                "producer." if constructible else
                "the named units did not satisfy the precheck"),
        },
        "ledger": {"llm": 0, "consumer_fits": 0, "downloads": 0},
    }
    FREEZE_V3_JSON.write_text(
        json.dumps(s1._plain(payload), ensure_ascii=False, indent=1,
                   sort_keys=True, default=str) + "\n", encoding="utf-8")
    FREEZE_V3_MD.write_text(_freeze_v3_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"]["verdict"],
        "course_name": selection["course_name"],
        "course": [row["unit_id"] for row in selection["course"]],
        "delta_material": selection["delta_material"],
        "artifact": str(FREEZE_V3_JSON),
    }, ensure_ascii=False, indent=1))
    return 0 if constructible else 1


# =========================================================================== #
# Part 0 v4 -- the final throw
# =========================================================================== #
COURSE_NAME_V4 = "discovery-and-support-reliable development curriculum"
FINAL_THROW_CAP = (
    "HARD CAP, written into the freeze: this is S1-v2's last course attempt.  "
    "A third empty treatment group is not a fourth reshuffle -- it is a "
    "systematic result.  If this run returns TREATMENT_EMPTY, everything "
    "stops and the mechanism goes back for review; no v5 is compiled."
)
# Third criterion, added by arbitration: a producer must not only propose the
# family (cold discovery) but also get it past the live Support gate under the
# half protocol.  v3 is where this became necessary: PowerCons proposed hampel
# and read +0.0357, which grades CONFLICT.
LIVE_SUPPORT_LEDGER = {
    "GunPointAgeSpan__impulse_v2": {
        "cold_discovery": "3/3", "live_support_pass": "3/3",
        "readings": "PS-0 re-earn +0.4000; S1-v2 v3 r1 +0.4500 POSITIVE",
        "role": "producer_A"},
    "GunPointMaleVersusFemale__impulse_v2": {
        "cold_discovery": "3/4", "live_support_pass": "3/4",
        "readings": "M-1 half protocol: supply conversion 2/4 -> 3/4 after "
                    "the wiring, +0.1867 deployed; S1-v2 v3 r1 A3/K0 both "
                    "earned +0.1867 on this unit",
        "role": "producer_B"},
    "GunPoint__impulse_v2": {
        "cold_discovery": "unmeasured", "live_support_pass": "unmeasured",
        "readings": "no live earn on record; carried as the backup producer "
                    "precisely because the other two are single points",
        "role": "producer_C_backup"},
    "PowerCons__impulse_v2": {
        "cold_discovery": "2/4", "live_support_pass": "0/2 at the material "
                                                     "line",
        "readings": "PS-0c +0.0714; S1-v2 v3 r1 +0.0357 -> CONFLICT.  Sealed "
                    "half-protocol margin is 5.00x, which the live readings "
                    "do not reproduce",
        "role": "beneficiary_weak"},
}
BACKUP_PRODUCER_SEMANTICS = (
    "Producer C is a third chance at the second positive, not a third "
    "positive.  The supply tier compiles as soon as any boundary holds two "
    "distinct unguided positives, so if A and B both land the card is written "
    "after position 3 and C still runs -- but by then the card is in A5's "
    "view, so C's own positive is Harness-conditioned and counts zero toward "
    "authorization.  That is the existing UNGUIDED rule, not an exception "
    "carved for this course, and it is why C cannot inflate the evidence."
)
FAMILY_NOTE_V4 = (
    "If the card compiles from A and B, its two Episodes are GunPointAgeSpan "
    "and GunPointMaleVersusFemale -- both GunPoint name family.  The strong "
    "beneficiary GPOvY is the same family again.  This is therefore a "
    "within-family transfer at the substrate level: development-mechanism "
    "grade, and it must not be reported as cross-family capability.  The weak "
    "beneficiary PowerCons is the one genuinely outside that family, which is "
    "part of why it is worth keeping despite its thin live Support."
)
POWERCONS_DISCREPANCY_NOTE = (
    "PowerCons__impulse_v2 carries a sealed half-protocol margin of 5.00x and "
    "live Support readings of +0.0714 (PS-0c) and +0.0357 (v3), the latter "
    "graded CONFLICT.  Sealed margin and live reading disagree, and the "
    "attribution is the proposal's parameter binding rather than the "
    "substrate: the sealed oracle scores the operator at its own tuned "
    "parameters, while the arm proposes it at whatever the contract binds.  "
    "Recorded as an honest weak stratum, not as a substrate defect."
)

PRODUCER_A_V4 = "GunPointAgeSpan__impulse_v2"
PRODUCER_B_V4 = "GunPointMaleVersusFemale__impulse_v2"
PRODUCER_C_V4 = "GunPoint__impulse_v2"
BENEFICIARY_STRONG_V4 = "GunPointOldVersusYoung__impulse_v2"
BENEFICIARY_WEAK_V4 = "PowerCons__impulse_v2"


def select_course_v4() -> dict[str, Any]:
    pool = _pool()
    by_id = {row["unit_id"]: row for row in pool}
    order = [
        (PRODUCER_A_V4, "producer_A"),
        ("BeetleFly__impulse_v2", "identity_A"),
        (PRODUCER_B_V4, "producer_B"),
        (PRODUCER_C_V4, "producer_C_backup"),
        (BENEFICIARY_STRONG_V4, "beneficiary_strong"),
        (BENEFICIARY_WEAK_V4, "beneficiary_weak"),
        ("Herring__impulse_v2", "heldout_only"),
        ("BirdChicken__burst_cls2", "identity_B"),
    ]
    missing = [unit for unit, _role in order if unit not in by_id]
    if missing:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "named units absent: %s" % ", ".join(missing))

    producers = [by_id[PRODUCER_A_V4], by_id[PRODUCER_B_V4]]
    scope = _scope_of(producers)
    pattern = dict((scope or {}).get("pattern_intersection") or {})
    ast = {"all": [{"feature": "task_kind", "op": "==",
                    "value": "classification"}]
           + [{"feature": key, "op": "==", "value": value}
              for key, value in sorted(pattern.items())]}

    beneficiaries = []
    for unit_id, band in ((BENEFICIARY_STRONG_V4, "strong"),
                          (BENEFICIARY_WEAK_V4, "weak")):
        row = by_id[unit_id]
        matched, _score = evaluate_applicability(
            ast, {"task_kind": "classification", **row["pattern"]})
        beneficiaries.append({
            "unit_id": unit_id, "margin_band": band,
            "machine_match_producer_scope": bool(matched),
            "census_learnability": row["census_learnability"],
            "half_margin": row["half_margin"],
            "half_meets_2x": row["half_meets_2x"],
            "material_line": _material_line(row),
            "n_slice_half_min": row["n_slice_half_min"],
            "live_support": LIVE_SUPPORT_LEDGER.get(unit_id, {}).get(
                "live_support_pass"),
            "is_also_a_producer": unit_id in (PRODUCER_A_V4, PRODUCER_B_V4,
                                              PRODUCER_C_V4),
        })

    course = []
    for position, (unit_id, role) in enumerate(order, start=1):
        row = by_id[unit_id]
        course.append({
            "position": position, "unit_id": unit_id, "role": role,
            "dataset": row["dataset"], "injection": row["injection"],
            "series_length": row["series_length"],
            "menu_oracle_program": row["menu_oracle_program"],
            "menu_oracle_heldout_utility": row["menu_oracle_heldout_utility"],
            "half_margin": row["half_margin"],
            "half_meets_2x": row["half_meets_2x"],
            "census_learnability": row["census_learnability"],
            "n_slice_half_min": row["n_slice_half_min"],
            "name_family": row["name_family"],
            "live_support_ledger": LIVE_SUPPORT_LEDGER.get(unit_id),
        })

    delta_parts = [row["material_line"] for row in beneficiaries]
    delta_material = (sum(part for part in delta_parts if part)
                      if all(delta_parts) else None)
    constructible = bool(
        all(row["machine_match_producer_scope"] for row in beneficiaries)
        and all(row["census_learnability"] == "LEARNABLE"
                for row in beneficiaries)
        and not any(row["is_also_a_producer"] for row in beneficiaries)
        and pattern)
    return {
        "course_name": COURSE_NAME_V4,
        "final_throw_cap": FINAL_THROW_CAP,
        "pool": pool,
        "producer_selection": {
            "rule": "three criteria: demonstrated cold discovery, live "
                    "Support pass rate under the half protocol, then sealed "
                    "margin; plus one backup producer",
            "third_criterion_origin": (
                "v3 showed cold discovery is not sufficient: PowerCons "
                "proposed hampel and read +0.0357, graded CONFLICT"),
            "ledger": LIVE_SUPPORT_LEDGER,
            "backup_producer_semantics": BACKUP_PRODUCER_SEMANTICS,
        },
        "exclusion_semantics_revision": EXCLUSION_SEMANTICS_V3,
        "family_overlap_note": FAMILY_NOTE_V4,
        "powercons_sealed_vs_live_note": POWERCONS_DISCREPANCY_NOTE,
        "natural_bootstrap_control": NATURAL_BOOTSTRAP_CONTROL,
        "rulings": {
            "a_beneficiaries_released": {
                "released": [BENEFICIARY_STRONG_V4, BENEFICIARY_WEAK_V4],
                "stratified_prediction": (
                    "pre-registered: A5's advantage should concentrate on the "
                    "strong-margin beneficiary (GPOvY) and be marginal or "
                    "absent on the weak one (PowerCons), whose live Support "
                    "has not cleared the material line in two attempts"),
                "prior_exposure_note": PRIOR_EXPOSURE_NOTE,
            },
            "b_regret_gate": {
                "definition": "sum of the two beneficiaries' half-protocol "
                              "material lines",
                "parts": {row["unit_id"]: row["material_line"]
                          for row in beneficiaries},
                "delta_material": delta_material,
                "cost_gate_unchanged": "convertible units average >= 1 probe "
                                       "saved",
            },
        },
        "producers_scope_v1": scope,
        "beneficiaries": beneficiaries,
        "course": course,
        "course_length": len(course),
        "transfer_graph": [{
            "from": [PRODUCER_A_V4, PRODUCER_B_V4,
                     PRODUCER_C_V4 + " (backup)"],
            "via": "first Slow boundary holding two distinct unguided "
                   "positives -> compile_supply_tier",
            "to": [BENEFICIARY_STRONG_V4, BENEFICIARY_WEAK_V4],
            "carrier": "supplies_candidates card (grants_execution=false)",
        }],
        "precheck": {
            "expected_card_boundary_after_position": "3 if A and B both land, "
                                                    "otherwise 4 via the "
                                                    "backup producer",
            "expected_first_divergence_position": 5,
            "expected_first_divergence_unit": BENEFICIARY_STRONG_V4,
            "five_axis_scope_non_empty": bool(pattern),
            "pattern_intersection_leaves": sorted(pattern),
            "beneficiaries_machine_match": all(
                row["machine_match_producer_scope"] for row in beneficiaries),
            "no_beneficiary_is_a_producer": not any(
                row["is_also_a_producer"] for row in beneficiaries),
            "k0_carries_no_card": "asserted at run time by compile_k0 purity",
        },
        "constructible": constructible,
        "delta_material": delta_material,
        "seeds": dict(SEEDS),
        "replicate_kind": "sampling",
        "replicate_semantics": (
            "the injection has no RNG to seed "
            "(run_e2_t6_cls_op_shared_harness.py:3896-3901), so a second run "
            "is a sampling replicate: identical substrate and protocol, Fast "
            "Agent the only stochastic element"),
        "protocol": {
            "slicing": "M-1 half protocol (one held-in round; Support = "
                       "concat(r1_support, r2_support), delayed = "
                       "concat(r1_delayed, r2_delayed))",
            "arms": [ARM_STATIC, ARM_A3, ARM_K0, ARM_A5],
            "k0": "bootstrap three cards + the inert Slow card; no "
                  "Target-local capability and no PS dual-source card",
            "llm_per_unit_per_arm": 12,
            "fit_per_unit_per_arm": 10,
            "llm_per_slow_boundary": 6,
            "llm_per_run_cap": 280,
        },
    }


def _freeze_v4_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# S1-v2 course freeze v4 -- %s" % payload["course_name"], "",
        "protocol: `%s`  git: `%s`" % (payload["protocol_version"],
                                       payload["git_head"]),
        "", "**%s**" % payload["verdict"]["verdict"], "",
        payload["verdict"]["reason"], "",
        "> **FINAL THROW.** %s" % payload["final_throw_cap"], "",
        "> **Exclusion semantics.** %s"
        % payload["exclusion_semantics_revision"], "",
        "> **Prior exposure.** %s"
        % payload["rulings"]["a_beneficiaries_released"][
            "prior_exposure_note"], "",
        "> **Family.** %s" % payload["family_overlap_note"], "",
        "> **Backup producer.** %s"
        % payload["producer_selection"]["backup_producer_semantics"], "",
        "> **PowerCons sealed vs live.** %s"
        % payload["powercons_sealed_vs_live_note"], "",
        "> **Control.** %s" % payload["natural_bootstrap_control"], "",
        "> **Replicates.** %s" % payload["replicate_semantics"], "",
        "## Course (frozen forward order)", "",
        "| # | role | unit | menu oracle | half margin | cold discovery | "
        "live Support pass |", "|---|---|---|---|---|---|---|",
    ]
    for row in payload["course"]:
        led = row.get("live_support_ledger") or {}
        lines.append("| %d | %s | `%s` | `%s` | %s | %s | %s |" % (
            row["position"], row["role"], row["unit_id"],
            row["menu_oracle_program"],
            ("%.2f" % row["half_margin"]) if row["half_margin"] else "-",
            led.get("cold_discovery", "-"),
            led.get("live_support_pass", "-")))
    lines += ["", "## Producer selection (third criterion)", "",
              "- rule: %s" % payload["producer_selection"]["rule"],
              "- why the third criterion exists: %s"
              % payload["producer_selection"]["third_criterion_origin"], "",
              "| unit | role | cold discovery | live Support pass | readings |",
              "|---|---|---|---|---|"]
    for unit, led in payload["producer_selection"]["ledger"].items():
        lines.append("| `%s` | %s | %s | %s | %s |" % (
            unit, led["role"], led["cold_discovery"],
            led["live_support_pass"], led["readings"]))
    lines += ["", "## Beneficiaries", "",
              "| unit | band | Scope match | census | half margin | material "
              "line | live Support | also a producer |",
              "|---|---|---|---|---|---|---|---|"]
    for row in payload["beneficiaries"]:
        lines.append("| `%s` | **%s** | %s | %s | %.2f | %.4f | %s | %s |" % (
            row["unit_id"], row["margin_band"],
            row["machine_match_producer_scope"], row["census_learnability"],
            row["half_margin"], row["material_line"], row["live_support"],
            row["is_also_a_producer"]))
    gate = payload["rulings"]["b_regret_gate"]
    lines += ["", "- stratified prediction: %s"
              % payload["rulings"]["a_beneficiaries_released"][
                  "stratified_prediction"], "",
              "## Gates", "",
              "- regret gate `Delta_material` = %s = %.6f"
              % (" + ".join("%.6f" % v for v in gate["parts"].values()),
                 gate["delta_material"]),
              "- cost gate: %s" % gate["cost_gate_unchanged"], "",
              "## Transfer graph", ""]
    for edge in payload["transfer_graph"]:
        lines.append("- %s --%s--> %s (carrier: %s)" % (
            " + ".join("`%s`" % unit for unit in edge["from"]), edge["via"],
            ", ".join("`%s`" % unit for unit in edge["to"]), edge["carrier"]))
    pre = payload["precheck"]
    lines += ["", "## Precheck", "",
              "- five-axis Scope non-empty: %s (%d leaves)"
              % (pre["five_axis_scope_non_empty"],
                 len(pre["pattern_intersection_leaves"])),
              "- both beneficiaries machine-match: %s"
              % pre["beneficiaries_machine_match"],
              "- no beneficiary is a producer: %s"
              % pre["no_beneficiary_is_a_producer"],
              "- expected card boundary: %s"
              % pre["expected_card_boundary_after_position"],
              "- expected first divergence: position %s (`%s`)"
              % (pre["expected_first_divergence_position"],
                 pre["expected_first_divergence_unit"]), ""]
    return "\n".join(lines) + "\n"


def freeze_v4() -> int:
    selection = select_course_v4()
    constructible = selection["constructible"]
    payload = {
        "protocol_version": PROTOCOL_VERSION + "_v4",
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "design": "docs/S1V2_DESIGN_DRAFT_2026-08-27.md",
        "supersedes": FREEZE_V3_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "arbitration": "three answers: sampling repeat authorized on SIGNAL, "
                       "live Support pass rate added as the third producer "
                       "criterion, one backup producer allowed.  Everything "
                       "else carries over from v3.",
        "sources": {
            "margins": PS0B_JSON.relative_to(PROJECT_ROOT).as_posix(),
            "oracles": "artifacts/functional/e2/s1_oracle/*.json "
                       "(exam keys only; not loaded into any arm)",
        },
        **selection,
        "verdict": {
            "verdict": ("S1V2_COURSE_FROZEN_V4" if constructible
                        else "COURSE_NOT_CONSTRUCTIBLE"),
            "reason": (
                "three producers -- two selected on cold discovery *and* live "
                "Support pass rate, plus one backup -- a non-empty five-axis "
                "Scope, and two held-in learnable beneficiaries at separated "
                "bands, neither of which is a producer.  Final throw."
                if constructible else
                "the named units did not satisfy the precheck"),
        },
        "ledger": {"llm": 0, "consumer_fits": 0, "downloads": 0},
    }
    FREEZE_V4_JSON.write_text(
        json.dumps(s1._plain(payload), ensure_ascii=False, indent=1,
                   sort_keys=True, default=str) + "\n", encoding="utf-8")
    FREEZE_V4_MD.write_text(_freeze_v4_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"]["verdict"],
        "course_name": selection["course_name"],
        "course": [row["unit_id"] for row in selection["course"]],
        "delta_material": selection["delta_material"],
        "artifact": str(FREEZE_V4_JSON),
    }, ensure_ascii=False, indent=1))
    return 0 if constructible else 1


# =========================================================================== #
# Part 1 -- the live four-arm forward course
# =========================================================================== #
SUPPLY_SKILL_ID = "s1v2_course_supply_v1"
LLM_PER_UNIT_PER_ARM = 12
FIT_PER_UNIT_PER_ARM = 10
LLM_PER_BOUNDARY = 6
LLM_TOTAL_CAP = 500
LLM_PER_RUN_CAP = 280
FIT_TOTAL_CAP = 900
WALL_SECONDS_CAP = int(6 * 60 * 60)
HALF_ROUNDS = ("r1",)


def _out_paths(seed: str) -> tuple[Path, Path, Path]:
    return (E2 / ("s1v2_v4_forward_run%s.json" % seed[-1]),
            E2 / ("s1v2_v4_forward_run%s.md" % seed[-1]),
            E2 / ("s1v2_v4_forward_run%s.checkpoint.json" % seed[-1]))


def _half_cell(quarter: Mapping[str, Any]) -> dict[str, Any]:
    """M-1's role-concat repack, generalised over units.

    One held-in round; Support and delayed each keep their own surface, so
    the dual gate is preserved and both confirmation surfaces double.
    """
    import run_e2_m1_margin_gate as m1

    surfaces = dict(quarter["surfaces"])
    support = m1._concat_surface(surfaces["r1_support"], surfaces["r2_support"])
    delayed = m1._concat_surface(surfaces["r1_delayed"], surfaces["r2_delayed"])
    half = dict(quarter)
    half["surfaces"] = {"r1_support": support, "r1_delayed": delayed}
    half["slice_rows"] = {"r1_support": int(support[0].shape[0]),
                          "r1_delayed": int(delayed[0].shape[0])}
    half["quarter_slice_rows"] = dict(quarter.get("slice_rows") or {})
    half["s1v2_protocol"] = "half_role_concat_one_round"
    return half


def _supply_rows_from(results: Sequence[Mapping[str, Any]],
                      *, card_installed_after: int | None) -> list[dict]:
    """A5's own earned Episodes, normalised for ``compile_supply_tier``.

    Earned means what PS-0 means by it: Support POSITIVE *and* the delayed
    gate confirmed it (LOCAL_ACTIVE).  An Episode produced after the course's
    own card was installed is marked conditioned, so it counts zero -- the
    card must not re-authorise itself.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        position = int(result.get("position") or 0)
        conditioned = bool(card_installed_after is not None
                           and position > card_installed_after)
        for record in result.get("rounds") or []:
            key = str(record.get("task_consumer_key") or "")
            if key.count("|") != 2:
                continue
            task_kind, consumer, metric = key.split("|")
            pattern = dict(record.get("fast_features_binned") or {})
            for episode in record.get("episodes") or []:
                if str(episode.get("relation")) != "POSITIVE":
                    continue
                if str(episode.get("local_status")) != "LOCAL_ACTIVE":
                    continue
                rows.append({
                    "task_episode_id": str(result["unit_id"]),
                    "unit_id": str(result["unit_id"]),
                    "run_id": "%s@%s" % (result["unit_id"],
                                         record.get("round")),
                    "program": str(episode.get("workflow_signature")),
                    "relation": "POSITIVE",
                    "conditioned_snapshot": conditioned,
                    "task_kind": task_kind, "consumer_id": consumer,
                    "metric": metric, "pattern": pattern,
                    "support_gain": float(episode.get("support_gain") or 0.0),
                    "delayed_gain": float(episode.get("delayed_gain") or 0.0),
                })
    return rows


def _score_unit(unit: Mapping[str, Any], arm: str,
                result: Mapping[str, Any]) -> dict[str, Any]:
    public = s1._public_unit_result(result)
    deployment = public.get("deployment") or {}
    deltas = deployment.get("heldout_recall_delta_by_class") or {}
    worst = min((float(v) for v in deltas.values()), default=0.0)
    utility = float(deployment.get("heldout_accuracy_gain") or 0.0)
    menu = float(unit.get("menu_oracle_heldout_utility") or 0.0)
    rounds = public.get("rounds") or []
    probes = sum(len(record.get("probes") or []) for record in rounds)
    supplied = [row for record in rounds
                for row in (record.get("proposals") or [])
                if str(row.get("candidate_id", "")).startswith("cand_skill_")]
    return {
        "position": unit["position"], "unit_id": unit["unit_id"],
        "role": unit["role"], "arm": arm,
        "deploy_source": deployment.get("deploy_source"),
        "applied_program": deployment.get("applied_program"),
        "applied_ops": [str(step.get("op"))
                        for step in (deployment.get("applied_program") or [])],
        "heldout_utility": utility,
        "menu_oracle_heldout_utility": menu,
        "regret": menu - utility,
        "heldout_recall_by_class": deployment.get("heldout_recall_by_class"),
        "worst_class_delta": worst,
        "harm_event": bool(worst < -cls.HARM_BAR),
        "llm_calls": int(result.get("llm_calls") or 0),
        "consumer_fits": int(result.get("consumer_fits") or 0),
        "probes": probes,
        "seconds": float(result.get("seconds") or 0.0),
        "approved_skill_ids": [record.get("approved_skill_id")
                               for record in rounds
                               if record.get("approved_skill_id")],
        "supply_candidates_in_pool": len(supplied),
        "supply_probed": sum(1 for row in supplied
                             if row.get("outcome") == "probe"),
        "agent_families": sorted({
            str(row.get("family")) for record in rounds
            for row in (record.get("proposals") or [])
            if row.get("family") not in (None, "identity")}),
        "rounds": rounds,
    }


def run_course(seed: str, *, resume: bool = False,
               finalize: bool = False) -> int:
    import run_e2_ps0c_ps1 as ps0c

    out_json, out_md, checkpoint = _out_paths(seed)
    frozen = json.loads(FREEZE_V4_JSON.read_text(encoding="utf-8"))
    course = list(frozen["course"])
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "seed": seed, "seed_label": SEEDS[seed],
        "replicate_kind": "sampling",
        "seed_ledger": (
            "The book asked for two forward runs on different injection "
            "seeds.  This family's injection has no RNG to seed -- "
            "run_e2_t6_cls_op_shared_harness.py:3896-3901 records it: a fixed "
            "signed template at positions derived from the series length, "
            "and a deterministic evenly-spaced fit/support split.  A 'fresh "
            "injection seed' would therefore be a fiction.  The two runs are "
            "honest *sampling* replicates: identical substrate and identical "
            "protocol, with the Fast Agent as the only stochastic element.  "
            "The seed label is a run id, not an injection parameter."),
        "course_source": FREEZE_V4_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "final_throw_cap": frozen.get("final_throw_cap"),
        "course_name": frozen.get("course_name"),
        "course": course,
        "delta_material": frozen["delta_material"],
        "rulings": frozen["rulings"],
        "protocol": frozen["protocol"],
        "prior_exposure_note": PRIOR_EXPOSURE_NOTE,
        "exclusion_semantics_revision": frozen.get(
            "exclusion_semantics_revision"),
        "family_overlap_note": frozen.get("family_overlap_note"),
        "natural_bootstrap_control": frozen.get("natural_bootstrap_control"),
        "analysis": (
            "ITT: a Scope-qualified unit whose supplied candidate failed to "
            "enter the pool counts as an A5 system failure.  The conditional "
            "conversion rate given successful injection is reported "
            "separately and is not the main analysis."),
    }
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    ledger = {"llm": 0, "fit": 0}
    done: set[tuple[int, str]] = set()
    if resume and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        rows = list(saved.get("rows") or [])
        events = list(saved.get("events") or [])
        ledger = {"llm": int((saved.get("ledger") or {}).get("llm") or 0),
                  "fit": int((saved.get("ledger") or {}).get("fit") or 0)}
        started = time.time() - float(saved.get("wall_seconds") or 0.0)
        done = {(int(row["position"]), str(row["arm"])) for row in rows}
        payload["resumed_from_checkpoint"] = sorted(
            "%s/%s" % (pos, arm) for pos, arm in done)

    def _save(stopped=None) -> None:
        checkpoint.write_text(json.dumps(ps0c.redact(s1._plain({
            "rows": rows, "events": events, "ledger": ledger,
            "wall_seconds": round(time.time() - started, 1),
        })), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    stopped: str | None = None
    if finalize:
        # Re-render this run's artifact from its own checkpoint.  No backend,
        # no new units, no rescoring: the rows are already persisted.
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        rows = list(saved.get("rows") or [])
        events = list(saved.get("events") or [])
        ledger = {"llm": int((saved.get("ledger") or {}).get("llm") or 0),
                  "fit": int((saved.get("ledger") or {}).get("fit") or 0)}
        started = time.time() - float(saved.get("wall_seconds") or 0.0)
        planned = {(int(row["position"]), arm) for row in course
                   for arm in (ARM_STATIC, ARM_A3, ARM_K0, ARM_A5)}
        got = {(int(row["position"]), str(row["arm"])) for row in rows}
        stopped = None if planned <= got else "COMPUTE_BUDGET_EXCEEDED"
        return _finish_run(payload, rows, events, ledger, frozen,
                           stopped=stopped, started=started,
                           out_json=out_json, out_md=out_md, ps0c=ps0c)
    try:
        probe = ps0c.probe_new_backend()
        payload["backend_probe"] = ps0c.redact(probe)
        print("PROBE ok=%s model=%s" % (probe.get("ok"),
                                        probe.get("returned_model")),
              flush=True)
        if not probe.get("ok"):
            raise Stop("BACKEND_UNAVAILABLE", str(probe.get("reason")))

        store_root = Path(tempfile.gettempdir()) / ("s1v2_%s" % seed)
        if not resume and store_root.exists():
            shutil.rmtree(store_root)
        k0 = s1.compile_k0(store_root)
        payload["k0"] = {"h0_sha": k0["h0_sha"], "k0_sha": k0["k0_sha"],
                         "purity": k0["purity"]}
        backend = cls._live_backend(LLM_PER_RUN_CAP)

        a5_snapshot = k0["k0"]
        a5_episodes: list[Any] = []
        a5_stamps: dict[str, str] = {}
        a5_results: list[dict[str, Any]] = []
        card_installed_after: int | None = None

        for unit in course:
            position = int(unit["position"])
            uid = str(unit["unit_id"])
            if time.time() - started > WALL_SECONDS_CAP:
                raise Stop("COMPUTE_BUDGET_EXCEEDED", "wall cap before " + uid)
            if ledger["llm"] >= LLM_PER_RUN_CAP or ledger["fit"] >= FIT_TOTAL_CAP:
                raise Stop("COMPUTE_BUDGET_EXCEEDED", "budget before " + uid)
            print("UNIT %d %s (%s)" % (position, uid, unit["role"]), flush=True)
            cell = _half_cell(s1._build_cell(unit))

            for arm in (ARM_STATIC, ARM_A3, ARM_K0, ARM_A5):
                if (position, arm) in done:
                    continue
                if arm == ARM_A5:
                    base, episodes, stamps = a5_snapshot, a5_episodes, a5_stamps
                elif arm == ARM_K0:
                    base, episodes, stamps = k0["k0"], (), {}
                else:
                    base, episodes, stamps = k0["h0"], (), {}
                result = s1.run_unit(
                    unit=unit, cell=cell, arm=arm, base_snapshot=base,
                    carried_episodes=episodes, agent_factory=cls._live_agent,
                    backend=backend, store_root=store_root,
                    rounds=HALF_ROUNDS, fit_cap=FIT_PER_UNIT_PER_ARM,
                    carried_stamps=stamps)
                ledger["llm"] = int(backend.calls)
                ledger["fit"] += int(result.get("consumer_fits") or 0)
                scored = _score_unit(unit, arm, result)
                rows.append(scored)
                _save()
                print("  %-10s deploy=%-34s gain=%+.4f regret=%+.4f "
                      "worst=%+.4f probes=%d llm=%d fit=%d"
                      % (arm, scored["deploy_source"],
                         scored["heldout_utility"], scored["regret"],
                         scored["worst_class_delta"], scored["probes"],
                         scored["llm_calls"], scored["consumer_fits"]),
                      flush=True)
                if arm == ARM_A5:
                    a5_episodes = list(result["_episodes"])
                    a5_stamps.update(
                        dict(result["_state"].get("domain_stamp") or {}))
                    a5_results.append({**scored, "position": position})

            # ---- Slow boundary: guard lifecycle already ran inside the
            # rounds; the supply tier is compiled here, mechanically.
            supply_rows = _supply_rows_from(
                a5_results, card_installed_after=card_installed_after)
            compiled = ss.compile_supply_tier(
                supply_rows, skill_id=SUPPLY_SKILL_ID,
                legal_features=ss._edit_schema_features(PROJECT_ROOT))
            event = {
                "after_position": position, "after_unit": uid,
                "supply_rows": len(supply_rows),
                "audit": compiled["audit"],
                "withheld_because": compiled["withheld_because"],
                "card_compiled": compiled["card"] is not None,
                "already_installed": card_installed_after is not None,
                "llm_calls": 0,
            }
            if compiled["card"] is not None and card_installed_after is None:
                a5_snapshot, _applied = s1._apply_entries(
                    a5_snapshot, [compiled["card"]],
                    store_root=store_root / "boundary",
                    tag="supply_%d" % position)
                card_installed_after = position
                event["installed"] = True
                event["runtime_bundle_sha"] = a5_snapshot.runtime_bundle_sha
                event["card_skill_id"] = compiled["card"]["skill_id"]
                event["scope_leaves"] = len(
                    compiled["card"]["observable_applicability"]["all"])
                print("  BOUNDARY supply card installed after position %d"
                      % position, flush=True)
            events.append(event)
            _save()
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
    return _finish_run(payload, rows, events, ledger, frozen,
                       stopped=stopped, started=started,
                       out_json=out_json, out_md=out_md, ps0c=ps0c)


def _finish_run(payload, rows, events, ledger, frozen, *, stopped, started,
                out_json, out_md, ps0c) -> int:
    payload["rows"] = rows
    payload["supply_events"] = events
    payload["summary"] = _summarise(rows, frozen["course"])
    payload["verdict"] = _run_verdict(rows, events, frozen, stopped=stopped)
    payload["ledger"] = {
        "llm": ledger["llm"], "llm_cap_per_run": LLM_PER_RUN_CAP,
        "fit": ledger["fit"], "fit_cap": FIT_TOTAL_CAP,
        "wall_seconds": round(time.time() - started, 1),
        "wall_seconds_cap": WALL_SECONDS_CAP, "downloads": 0,
    }
    payload["obligations"] = _obligations()
    out_json.write_text(json.dumps(
        ps0c.redact(s1._plain(payload)), ensure_ascii=False, indent=1,
        sort_keys=True, default=str) + "\n", encoding="utf-8")
    out_md.write_text(_run_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "reason": payload["verdict"].get("reason"),
                      "ledger": payload["ledger"],
                      "artifact": str(out_json)},
                     ensure_ascii=False, indent=1), flush=True)
    return 0


def _summarise(rows, course) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in (ARM_STATIC, ARM_A3, ARM_K0, ARM_A5):
        arm_rows = [row for row in rows if row["arm"] == arm]
        out[arm] = {
            "units": len(arm_rows),
            "cumulative_regret": sum(row["regret"] for row in arm_rows),
            "mean_heldout_utility": (
                sum(row["heldout_utility"] for row in arm_rows)
                / len(arm_rows)) if arm_rows else 0.0,
            "worst_class_min": min((row["worst_class_delta"]
                                    for row in arm_rows), default=0.0),
            "harm_events": sum(1 for row in arm_rows if row["harm_event"]),
            "llm": sum(row["llm_calls"] for row in arm_rows),
            "consumer_fits": sum(row["consumer_fits"] for row in arm_rows),
            "probes": sum(row["probes"] for row in arm_rows),
            "fit_wall_seconds": sum(row["seconds"] for row in arm_rows),
            "deployed_non_identity": sum(
                1 for row in arm_rows if row["applied_ops"]),
            "supply_candidates_in_pool": sum(
                row["supply_candidates_in_pool"] for row in arm_rows),
        }
    return out


def _run_verdict(rows, events, frozen, *, stopped) -> dict[str, Any]:
    summary = _summarise(rows, frozen["course"])
    a5, a3, k0 = summary[ARM_A5], summary[ARM_A3], summary[ARM_K0]
    delta = float(frozen["delta_material"])
    beneficiaries = {row["unit_id"] for row in frozen["course"]
                     if row["role"].startswith("beneficiary")}
    a5_benefit_rows = [row for row in rows
                       if row["arm"] == ARM_A5
                       and row["unit_id"] in beneficiaries]
    injected = sum(1 for row in a5_benefit_rows
                   if row["supply_candidates_in_pool"])
    any_knowledge = bool(
        any(event.get("card_compiled") for event in events)
        or any(row["supply_candidates_in_pool"] for row in rows))
    facts = {
        "summary": summary,
        "delta_material": delta,
        "beneficiary_units": sorted(beneficiaries),
        "beneficiaries_with_injection": injected,
        "beneficiaries_total": len(a5_benefit_rows),
        "regret_gap_vs_a3": a3["cumulative_regret"] - a5["cumulative_regret"],
        "regret_gap_vs_k0": k0["cumulative_regret"] - a5["cumulative_regret"],
        "probe_gap_vs_a3": a3["probes"] - a5["probes"],
        "quality_gap_vs_a3": (a5["mean_heldout_utility"]
                              - a3["mean_heldout_utility"]),
        "quality_gap_vs_k0": (a5["mean_heldout_utility"]
                              - k0["mean_heldout_utility"]),
        "harm_a5": a5["harm_events"], "harm_a3": a3["harm_events"],
        "harm_k0": k0["harm_events"],
        "card_installed": [event for event in events
                           if event.get("installed")],
    }
    if stopped:
        return {"verdict": stopped, "facts": facts,
                "reason": "stopped before the full course"}
    if not any_knowledge:
        return {"verdict": "TREATMENT_EMPTY", "facts": facts,
                "reason": ("the course produced no Fast-visible knowledge: no "
                           "supply card compiled and no supplied candidate "
                           "ever reached a pool.  Stop; the second seed is "
                           "not started.")}
    non_inferior = (facts["quality_gap_vs_a3"] >= -0.005
                    and facts["quality_gap_vs_k0"] >= -0.005
                    and a5["worst_class_min"] >= min(
                        a3["worst_class_min"], k0["worst_class_min"]) - 0.005
                    and a5["harm_events"] <= min(a3["harm_events"],
                                                 k0["harm_events"]))
    regret_gate = (facts["regret_gap_vs_a3"] >= delta
                   and facts["regret_gap_vs_k0"] >= delta)
    cost_gate = (len(a5_benefit_rows) > 0
                 and facts["probe_gap_vs_a3"] >= len(a5_benefit_rows))
    attributable = bool(facts["card_installed"] and injected)
    if non_inferior and (regret_gate or cost_gate) and attributable:
        return {"verdict": "S1V2_FORWARD_SIGNAL", "facts": facts,
                "reason": ("A5-online is non-inferior on quality and harm and "
                           "clears a material gate, and the advantage traces "
                           "to a card the course compiled from its own "
                           "Episodes.")}
    if not attributable and any_knowledge:
        return {"verdict": "NO_TRANSFER", "facts": facts,
                "reason": ("knowledge was produced but never reached a "
                           "beneficiary pool, so nothing could transfer.")}
    if facts["quality_gap_vs_a3"] < -0.005 or a5["harm_events"] > a3["harm_events"]:
        return {"verdict": "NEGATIVE_TRANSFER", "facts": facts,
                "reason": ("A5-online is worse than A3-reset on quality or "
                           "harm; first fault is the supplied candidate "
                           "displacing a better local choice.")}
    return {"verdict": "NO_TRANSFER", "facts": facts,
            "reason": ("the card reached the beneficiaries but neither the "
                       "regret nor the cost gate cleared.")}


def _obligations() -> dict[str, Any]:
    return {
        "course_frozen_before_any_live_run": True,
        "thresholds_and_authorization_unmodified": (
            "MATERIAL, the TRY tier's leave-one-out, the supply tier's count, "
            "the T1 predicate and the ledger incumbent rule are untouched"),
        "methods_package_unmodified": True,
        "runtime_contracts_operators_unmodified": True,
        "k0_carries_no_dual_source_card": True,
        "a5_knowledge_is_course_produced_only": True,
        "oracle_read_as_exam_key_only": True,
        "guided_positive_counts_zero": True,
        "itt_main_analysis": True,
        "downloads": 0,
        "full_repo_pytest_not_run": True,
    }


def _run_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = [
        "# S1-v2 forward course, seed %s" % payload["seed"], "",
        "protocol: `%s`  git: `%s`  run label: `%s` (%s replicate)"
        % (payload["protocol_version"], payload["git_head"],
           payload["seed_label"], payload.get("replicate_kind", "sampling")),
        "", "**%s**" % verdict["verdict"], "", verdict.get("reason", ""), "",
        "> %s" % payload["seed_ledger"], "",
        "> %s" % payload["prior_exposure_note"], "",
        "> %s" % payload["analysis"], "",
        "## Per-unit, per-arm", "",
        "| # | role | unit | arm | deployed | held-out | regret | worst class "
        "| probes | LLM | fits |", "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append("| %s | %s | %s | %s | `%s` | %+.4f | %+.4f | %+.4f | %d "
                     "| %d | %d |" % (
                         row["position"], row["role"],
                         row["unit_id"].split("__")[0], row["arm"],
                         ",".join(row["applied_ops"]) or "identity",
                         row["heldout_utility"], row["regret"],
                         row["worst_class_delta"], row["probes"],
                         row["llm_calls"], row["consumer_fits"]))
    lines += ["", "## Arm summary", "",
              "| arm | units | cumulative regret | mean held-out | worst class "
              "| harm | probes | LLM | fits | fit wall (s) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, row in payload["summary"].items():
        lines.append("| %s | %d | %+.4f | %+.4f | %+.4f | %d | %d | %d | %d | "
                     "%.1f |" % (
                         arm, row["units"], row["cumulative_regret"],
                         row["mean_heldout_utility"], row["worst_class_min"],
                         row["harm_events"], row["probes"], row["llm"],
                         row["consumer_fits"], row["fit_wall_seconds"]))
    lines += ["", "## Supply / guard timeline", "",
              "| after # | unit | rows | card compiled | installed | withheld "
              "because |", "|---|---|---|---|---|---|"]
    for event in payload["supply_events"]:
        lines.append("| %s | %s | %d | %s | %s | %s |" % (
            event["after_position"], event["after_unit"].split("__")[0],
            event["supply_rows"], event["card_compiled"],
            event.get("installed", False),
            event.get("withheld_because") or "-"))
    facts = verdict.get("facts") or {}
    ledger = payload["ledger"]
    lines += ["", "## Gates", "",
              "- Delta_material = %.6f" % payload["delta_material"],
              "- regret gap vs A3-reset = %+.4f; vs K0-fixed = %+.4f"
              % (facts.get("regret_gap_vs_a3", 0.0),
                 facts.get("regret_gap_vs_k0", 0.0)),
              "- probe gap vs A3-reset = %s"
              % facts.get("probe_gap_vs_a3"),
              "- beneficiaries with an injected candidate: %s / %s"
              % (facts.get("beneficiaries_with_injection"),
                 facts.get("beneficiaries_total")),
              "", "## Cost", "",
              "- LLM: %s / %s" % (ledger["llm"], ledger["llm_cap_per_run"]),
              "- fits: %s / %s" % (ledger["fit"], ledger["fit_cap"]),
              "- wall: %s s / %s s" % (ledger["wall_seconds"],
                                       ledger["wall_seconds_cap"]),
              "- downloads: 0", "", "## Obligations", ""]
    for key, value in payload["obligations"].items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


# =========================================================================== #
# L1 -- ladder revision v2 replay from the v4 boundary
# =========================================================================== #
V4_CHECKPOINT = E2 / "s1v2_v4_forward_run1.checkpoint.json"
L1_SKILL_ID = "l1_ladder_v2_supply_v1"
L1_BOUNDARY_POSITION = 3
L1_TAIL_POSITIONS = (4, 5, 6, 7, 8)
L1_LLM_CAP = 120
L1_FIT_CAP = 300
L1_WALL_CAP = int(4 * 60 * 60)


def _l1_paths(seed: str) -> tuple[Path, Path, Path]:
    return (E2 / ("l1_ladder_v2_replay_%s.json" % seed),
            E2 / ("l1_ladder_v2_replay_%s.md" % seed),
            E2 / ("l1_ladder_v2_replay_%s.checkpoint.json" % seed))


def _v4_rows() -> list[dict[str, Any]]:
    saved = json.loads(V4_CHECKPOINT.read_text(encoding="utf-8"))
    return list(saved.get("rows") or [])


def _line_of(path: str, needle: str) -> str:
    text = (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines()
    line = next((i + 1 for i, t in enumerate(text) if needle in t), None)
    return "%s:%s" % (path, line)


def _l1_boundary_card() -> tuple[dict[str, Any] | None, dict[str, Any],
                                 list[dict[str, Any]]]:
    """Recompile the unit-3 boundary at the new price, off the v4 record."""
    rows = [row for row in _v4_rows()
            if row["arm"] == ARM_A5 and int(row["position"]) <= L1_BOUNDARY_POSITION]
    supply_rows = _supply_rows_from(rows, card_installed_after=None)
    compiled = ss.compile_supply_tier(
        supply_rows, skill_id=L1_SKILL_ID,
        legal_features=ss._edit_schema_features(PROJECT_ROOT))
    return compiled["card"], compiled, supply_rows


def _l1_t1(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Six offline checks.  Zero LLM, zero Consumer fits."""
    import dataclasses

    from SelfEvolvingHarnessTS.contracts.harness import load_skill_entry
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _supply_rung_candidates,
    )
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
        extract_public_features,
    )
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
        _is_inert_experience_card,
        resolve_harness_view,
    )
    from SelfEvolvingHarnessTS.runtime.candidate_verification import (
        verify_candidate,
    )
    import numpy as np

    checks: list[dict[str, Any]] = []

    # (1) the price constant, and proof the TRY tier was not touched.
    checks.append({
        "check": "1. supply-tier price constant is 1",
        "pass": ss.SUPPLY_TIER_MIN_DISTINCT_TASKS == 1,
        "evidence": {
            "constant": ss.SUPPLY_TIER_MIN_DISTINCT_TASKS,
            "file_line": _line_of(
                "evaluation/functional/task_episode_harness/agentic/"
                "source_skill.py", "SUPPLY_TIER_MIN_DISTINCT_TASKS = 1"),
            "try_tier_untouched_file_line": _line_of(
                "evaluation/functional/task_episode_harness/agentic/"
                "source_skill.py", "loo_minimum = ("),
            "try_tier_still_loo": "authorization_audit computes "
                                  "leave_one_out_minimum_positive and gates "
                                  "active_try_authorized on it; no edit in "
                                  "this book touches that function",
        },
    })

    # (2) the boundary compiles a single-Episode card, deterministically.
    card, compiled, supply_rows = _l1_boundary_card()
    twice = ss.compile_supply_tier(
        supply_rows, skill_id=L1_SKILL_ID,
        legal_features=ss._edit_schema_features(PROJECT_ROOT))["card"]
    authority = (card or {}).get("risk_guards", {}).get("authority") or {}
    scope = (card or {}).get("risk_guards", {}).get("scope_v1") or {}
    evidence = (card or {}).get("risk_guards", {}).get("evidence") or {}
    checks.append({
        "check": "2. unit-3 boundary compiles the single-Episode card",
        "pass": bool(
            card is not None
            and authority == {"reorders_supplied_candidates": False,
                              "supplies_candidates": True,
                              "suppresses_operators": False,
                              "grants_execution": False}
            and int(evidence.get("source_count") or 0) == 1
            and json.dumps(card, sort_keys=True)
            == json.dumps(twice, sort_keys=True)),
        "evidence": {
            "supply_rows": supply_rows,
            "audit": compiled["audit"],
            "authority": authority,
            "scope_v1": scope,
            "evidence_block": evidence,
            "deterministic_recompile": json.dumps(card, sort_keys=True)
            == json.dumps(twice, sort_keys=True),
            "dual_gate_enforced_at": _line_of(
                "evaluation/functional/run_e2_s1v2_forward_course.py",
                'if str(episode.get("local_status")) != "LOCAL_ACTIVE"'),
        },
    })
    if card is None:
        return {"checks": checks, "pass": False, "card": None,
                "first_fault": "boundary did not compile a card at price 1"}

    # (3) the inert predicate does not withhold it from Fast.
    entry = load_skill_entry(card)
    checks.append({
        "check": "3. T1 inert predicate does not withhold the supply card",
        "pass": _is_inert_experience_card(entry) is False,
        "evidence": {
            "is_inert": _is_inert_experience_card(entry),
            "why": "the predicate only classifies six-section experience "
                   "cards; a supply card carries no risk_guards.sections, so "
                   "_experience_card_sections returns None and the predicate "
                   "returns False before any clause is read",
            "predicate_file_line": _line_of(
                "methods/ttha/retrieval.py", "def _is_inert_experience_card"),
            "sections_probe_file_line": _line_of(
                "methods/ttha/retrieval.py",
                "def _experience_card_sections"),
            "carve_out_needed": False,
            "methods_edits_this_book": 0,
        },
    })

    # (4) Scope match over the tail units' frozen pattern views.
    h0 = compile_snapshot(
        PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
        verify_lock=False)
    snapshot = dataclasses.replace(h0, skills=(*h0.skills, entry))
    tail = [row for row in frozen["course"]
            if int(row["position"]) in L1_TAIL_POSITIONS]
    table = []
    for row in tail:
        oracle = _oracle(str(row["unit_id"])) or {}
        features = {"task_kind": "classification",
                    **dict(oracle.get("public_features_binned") or {})}
        matched, _score = evaluate_applicability(
            card["observable_applicability"], features)
        view = resolve_harness_view(snapshot, features, role="fast")
        table.append({
            "position": row["position"], "unit_id": row["unit_id"],
            "role": row["role"], "machine_match": bool(matched),
            "served_in_fast_view": L1_SKILL_ID in view.skill_ids,
        })
    predicted_match = {"GunPoint__impulse_v2",
                       "GunPointOldVersusYoung__impulse_v2",
                       "PowerCons__impulse_v2", "Herring__impulse_v2"}
    checks.append({
        "check": "4. Scope match precheck over the tail five units",
        "pass": any(row["machine_match"] for row in table),
        "evidence": {
            "table": table,
            "matched": sorted(row["unit_id"] for row in table
                              if row["machine_match"]),
            "pre_registered_match": sorted(predicted_match),
            "matches_prediction": (
                {row["unit_id"] for row in table if row["machine_match"]}
                == predicted_match),
            "stop_rule": "if none of the five matched this is "
                         "SCOPE_TOO_NARROW_TO_MATTER and the book stops; "
                         "widening Scope on the spot is forbidden",
        },
    })
    if not any(row["machine_match"] for row in table):
        return {"checks": checks, "pass": False, "card": card,
                "first_fault": "SCOPE_TOO_NARROW_TO_MATTER"}

    # (5) injection dry run on the first matching tail unit.
    target = next(row for row in table if row["machine_match"])
    unit = next(row for row in frozen["course"]
                if str(row["unit_id"]) == target["unit_id"])
    cell = _half_cell(s1._build_cell(unit))
    block = np.asarray(cell["observation_block"], dtype=np.float64)
    features = dict(extract_public_features(block, task_kind="classification"))
    view = resolve_harness_view(snapshot, features, role="fast")
    supplied = _supply_rung_candidates(view, features)
    receipts = []
    for candidate in supplied:
        artifact = verify_candidate(
            candidate, block,
            allowed_operators=[op for op, _p
                               in candidate.program.execution_steps()],
            inspected_regions=(),
            maximum_modified_fraction=float(
                cls._task_context().deployment_constraints
                .maximum_modified_fraction),
            preserve_outside_inspected_region=True,
            require_finite_output=True)
        receipts.append({
            "candidate_id": candidate.candidate_id,
            "status": artifact.receipt.status,
            "selectable": artifact.selectable,
            "modified_fraction": artifact.receipt.modified_fraction,
            "rejection_code": artifact.receipt.rejection_code,
        })
    checks.append({
        "check": "5. injection dry run materialises and verifies",
        "pass": bool(supplied and all(row["selectable"] for row in receipts)),
        "evidence": {
            "unit_id": target["unit_id"],
            "served": L1_SKILL_ID in view.skill_ids,
            "candidate_ids": [c.candidate_id for c in supplied],
            "verifier_receipts": receipts,
            "reader_file_line": _line_of("methods/ttha/fast_agent.py",
                                         "def _supply_rung_candidates"),
            "probe_slot_semantics": "W-1: the supplied candidate occupies a "
                                    "slot inside maximum_candidates and does "
                                    "not delete the agent's exploration slot",
        },
    })

    # (6) a positive earned while the card is in view counts zero.
    guided = _supply_rows_from(
        [row for row in _v4_rows()
         if row["arm"] == ARM_A5 and int(row["position"]) <= L1_BOUNDARY_POSITION],
        card_installed_after=L1_BOUNDARY_POSITION - 1)
    guided_audit = ss.supply_tier_audit(guided)
    checks.append({
        "check": "6. guided positives count zero",
        "pass": bool(guided_audit
                     and guided_audit[0]["unguided_positive"] == 0
                     and guided_audit[0]["conditioned_positive"] >= 1),
        "evidence": {
            "audit_when_marked_conditioned": guided_audit,
            "marker_file_line": _line_of(
                "evaluation/functional/run_e2_s1v2_forward_course.py",
                "conditioned = bool(card_installed_after is not None"),
            "rule_file_line": _line_of(
                "evaluation/functional/task_episode_harness/agentic/"
                "source_skill.py",
                'key = ("positive_conditioned" if row.get(conditioning_key)'),
            "consequence": "a single-Episode card cannot license its own "
                           "promotion or a wider Scope",
        },
    })
    return {"checks": checks, "pass": all(row["pass"] for row in checks),
            "card": card, "scope_table": table,
            "first_fault": next((row["check"] for row in checks
                                 if not row["pass"]), None)}


def l1_replay(seed: str = "r1", *, t1_only: bool = False,
              resume: bool = False) -> int:
    import run_e2_ps0c_ps1 as ps0c

    out_json, out_md, checkpoint = _l1_paths(seed)
    frozen = json.loads(FREEZE_V4_JSON.read_text(encoding="utf-8"))
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)
    payload: dict[str, Any] = {
        "protocol_version": "l1_ladder_v2_replay_v1",
        "evidence_grade": "development",
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "seed": seed,
        "ladder_revision": (
            "v2: supply-tier evidence price 2 -> 1 strong positive (Support "
            "and delayed both POSITIVE).  TRY tier LOO, RISK tier, execution "
            "and deployment gates, MATERIAL and the prompt/model/budget "
            "protocol are all untouched."),
        "course_source": FREEZE_V4_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "v4_source": V4_CHECKPOINT.relative_to(PROJECT_ROOT).as_posix(),
        "course": frozen["course"],
        "delta_material": frozen["delta_material"],
        "pre_registered_predictions": {
            "card_compiles_at_boundary": True,
            "scope_matches": ["GunPoint__impulse_v2",
                              "GunPointOldVersusYoung__impulse_v2",
                              "PowerCons__impulse_v2", "Herring__impulse_v2"],
            "scope_does_not_match": ["BirdChicken__burst_cls2"],
            "converts": ["GunPoint__impulse_v2",
                         "GunPointOldVersusYoung__impulse_v2"],
            "does_not_convert": ["PowerCons__impulse_v2"],
            "abstains": ["Herring__impulse_v2"],
            "harm_events": 0,
            "a5_tail_regret": "from +0.7710 down to <= 0.20",
        },
    }
    payload["t1"] = _l1_t1(frozen)
    if not payload["t1"]["pass"] or t1_only:
        payload["verdict"] = {
            "verdict": ("L1_T1_PASSED_NO_LIVE" if payload["t1"]["pass"]
                        else "L1_T1_FAILED"),
            "reason": (payload["t1"].get("first_fault")
                       or "offline gate passed; live replay not requested"),
        }
        payload["ledger"] = {"llm": 0, "fit": 0, "downloads": 0,
                             "wall_seconds": round(time.time() - started, 1)}
        payload["rows"] = []
        out_json.write_text(json.dumps(
            ps0c.redact(s1._plain(payload)), ensure_ascii=False, indent=1,
            sort_keys=True, default=str) + "\n", encoding="utf-8")
        out_md.write_text(_l1_markdown(payload), encoding="utf-8")
        print(json.dumps({"verdict": payload["verdict"]["verdict"],
                          "t1_pass": payload["t1"]["pass"],
                          "artifact": str(out_json)},
                         ensure_ascii=False, indent=1))
        return 0 if payload["t1"]["pass"] else 1

    card = payload["t1"]["card"]
    rows: list[dict[str, Any]] = []
    ledger = {"llm": 0, "fit": 0}
    if resume and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        rows = list(saved.get("rows") or [])
        ledger = {"llm": int((saved.get("ledger") or {}).get("llm") or 0),
                  "fit": int((saved.get("ledger") or {}).get("fit") or 0)}
        started = time.time() - float(saved.get("wall_seconds") or 0.0)
    done = {int(row["position"]) for row in rows}

    stopped: str | None = None
    try:
        probe = ps0c.probe_new_backend()
        payload["backend_probe"] = ps0c.redact(probe)
        print("PROBE ok=%s model=%s" % (probe.get("ok"),
                                        probe.get("returned_model")),
              flush=True)
        if not probe.get("ok"):
            raise Stop("BACKEND_UNAVAILABLE", str(probe.get("reason")))
        store_root = Path(tempfile.gettempdir()) / ("l1_%s" % seed)
        if not resume and store_root.exists():
            shutil.rmtree(store_root)
        k0 = s1.compile_k0(store_root)
        a5_snapshot, _applied = s1._apply_entries(
            k0["k0"], [card], store_root=store_root / "boundary",
            tag="l1_supply")
        payload["boundary_resume"] = {
            "kind": "boundary_replay",
            "carried": "the supply card compiled from the recorded unit-3 "
                       "Episode, installed on K0 through the frozen edit path",
            "not_carried": (
                "A5's in-memory Episode objects and its unit-3 Target-local "
                "capability.  The Episode rows survive as the card's evidence "
                "block; the Target-local Skill is domain-stamped and could "
                "not apply to any tail unit anyway.  Stated so the attribution "
                "is not overclaimed."),
            "producer_stage_not_rerun": True,
            "runtime_bundle_sha": a5_snapshot.runtime_bundle_sha,
            "k0_sha": k0["k0_sha"],
        }
        backend = cls._live_backend(L1_LLM_CAP)
        for unit in frozen["course"]:
            position = int(unit["position"])
            if position not in L1_TAIL_POSITIONS or position in done:
                continue
            if time.time() - started > L1_WALL_CAP:
                raise Stop("COMPUTE_BUDGET_EXCEEDED", "wall cap")
            if ledger["llm"] >= L1_LLM_CAP or ledger["fit"] >= L1_FIT_CAP:
                raise Stop("COMPUTE_BUDGET_EXCEEDED", "budget cap")
            print("UNIT %d %s (%s)" % (position, unit["unit_id"],
                                       unit["role"]), flush=True)
            cell = _half_cell(s1._build_cell(unit))
            result = s1.run_unit(
                unit=unit, cell=cell, arm=ARM_A5, base_snapshot=a5_snapshot,
                carried_episodes=(), agent_factory=cls._live_agent,
                backend=backend, store_root=store_root,
                rounds=HALF_ROUNDS, fit_cap=FIT_PER_UNIT_PER_ARM,
                carried_stamps={})
            ledger["llm"] = int(backend.calls)
            ledger["fit"] += int(result.get("consumer_fits") or 0)
            scored = _score_unit(unit, ARM_A5, result)
            scored["candidate_sources"] = _candidate_sources(scored)
            rows.append(scored)
            checkpoint.write_text(json.dumps(ps0c.redact(s1._plain({
                "rows": rows, "ledger": ledger,
                "wall_seconds": round(time.time() - started, 1)})),
                indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            src = scored["candidate_sources"]
            print("  A5 deploy=%-34s gain=%+.4f regret=%+.4f worst=%+.4f "
                  "supplied=%d self=%d probes=%d llm=%d fit=%d"
                  % (scored["deploy_source"], scored["heldout_utility"],
                     scored["regret"], scored["worst_class_delta"],
                     src["supplied_in_pool"], src["self_proposed_in_pool"],
                     scored["probes"], scored["llm_calls"],
                     scored["consumer_fits"]), flush=True)
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
    payload["rows"] = rows
    payload["v4_control"] = _v4_tail_control()
    payload["verdict"] = _l1_verdict(rows, payload, stopped=stopped)
    payload["prediction_table"] = _l1_prediction_table(payload, rows)
    payload["ledger"] = {
        "llm": ledger["llm"], "llm_cap": L1_LLM_CAP,
        "fit": ledger["fit"], "fit_cap": L1_FIT_CAP,
        "wall_seconds": round(time.time() - started, 1),
        "wall_seconds_cap": L1_WALL_CAP, "downloads": 0,
    }
    payload["obligations"] = _l1_obligations()
    out_json.write_text(json.dumps(
        ps0c.redact(s1._plain(payload)), ensure_ascii=False, indent=1,
        sort_keys=True, default=str) + "\n", encoding="utf-8")
    out_md.write_text(_l1_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "headline": payload["verdict"].get("headline"),
                      "ledger": payload["ledger"],
                      "artifact": str(out_json)},
                     ensure_ascii=False, indent=1), flush=True)
    return 0


def _candidate_sources(scored: Mapping[str, Any]) -> dict[str, Any]:
    supplied = self_proposed = 0
    supplied_probed = self_probed = 0
    supplied_won = False
    for record in scored.get("rounds") or []:
        for row in record.get("proposals") or []:
            cid = str(row.get("candidate_id") or "")
            if cid == "identity":
                continue
            if cid.startswith("cand_skill_"):
                supplied += 1
                supplied_probed += int(row.get("outcome") == "probe")
            else:
                self_proposed += 1
                self_probed += int(row.get("outcome") == "probe")
        winner = record.get("winner_program") or []
        if winner and record.get("approved_skill_id"):
            supplied_won = supplied_won or bool(
                any(str(p.get("candidate_id", "")).startswith("cand_skill_")
                    and p.get("chosen_by_select") is not None
                    for p in (record.get("proposals") or [])))
    return {"supplied_in_pool": supplied,
            "self_proposed_in_pool": self_proposed,
            "supplied_probed": supplied_probed,
            "self_proposed_probed": self_probed,
            "supplied_reached_winner": supplied_won}


def _v4_tail_control() -> dict[str, Any]:
    rows = [row for row in _v4_rows()
            if int(row["position"]) in L1_TAIL_POSITIONS]
    out: dict[str, Any] = {"note": (
        "v4 frozen readings, used as a replay-grade control.  The other three "
        "arms are not re-run in L1, so these are not a fresh contemporaneous "
        "comparison and must not be reported as one.")}
    for arm in (ARM_STATIC, ARM_A3, ARM_K0, ARM_A5):
        arm_rows = [row for row in rows if row["arm"] == arm]
        out[arm] = {
            "units": len(arm_rows),
            "cumulative_regret": sum(row["regret"] for row in arm_rows),
            "harm_events": sum(1 for row in arm_rows
                               if row.get("harm_event")),
            "probes": sum(row["probes"] for row in arm_rows),
            "llm": sum(row["llm_calls"] for row in arm_rows),
            "consumer_fits": sum(row["consumer_fits"] for row in arm_rows),
        }
    return out


def _l1_verdict(rows, payload, *, stopped) -> dict[str, Any]:
    control = _v4_tail_control()
    a5_v4 = control[ARM_A5]["cumulative_regret"] if control.get(ARM_A5) else 0.0
    regret = sum(row["regret"] for row in rows)
    harm = sum(1 for row in rows if row["harm_event"])
    converted = [row for row in rows
                 if row["candidate_sources"]["supplied_probed"]
                 and row["applied_ops"]]
    supplied_units = [row for row in rows
                      if row["candidate_sources"]["supplied_in_pool"]]
    gain = a5_v4 - regret
    delta = float(payload["delta_material"])
    headline = ("YES, %+.4f cumulative regret against the v4 A5 tail "
                "(%.4f -> %.4f), gate %.6f"
                % (gain, a5_v4, regret, delta)) if gain >= delta else (
        "NO, %+.4f cumulative regret against the v4 A5 tail (%.4f -> %.4f), "
        "gate %.6f" % (gain, a5_v4, regret, delta))
    facts = {
        "a5_tail_regret_v4": a5_v4, "a5_tail_regret_l1": regret,
        "regret_improvement": gain, "delta_material": delta,
        "harm_events": harm,
        "units_with_supplied_candidate": len(supplied_units),
        "units_converted_from_supply": [row["unit_id"] for row in converted],
        "units_run": len(rows),
    }
    if stopped:
        return {"verdict": stopped, "headline": headline, "facts": facts,
                "reason": "stopped before the tail completed"}
    if not supplied_units:
        return {"verdict": "L1_NO_INJECTION", "headline": headline,
                "facts": facts,
                "reason": ("the card compiled but no tail unit received the "
                           "supplied candidate in its pool")}
    if not converted:
        return {"verdict": "L1_NO_CONVERSION", "headline": headline,
                "facts": facts,
                "reason": ("the supplied candidate reached %d pool(s) and "
                           "converted nowhere" % len(supplied_units))}
    if harm == 0 and gain >= delta:
        return {"verdict": "L1_SIGNAL", "headline": headline, "facts": facts,
                "reason": ("supply-sourced conversion produced a material "
                           "regret improvement with zero harm")}
    return {"verdict": "L1_CONVERSION_BELOW_GATE", "headline": headline,
            "facts": facts,
            "reason": ("conversion happened but the improvement did not "
                       "clear the material gate, or harm was not zero")}


def _l1_prediction_table(payload, rows) -> list[dict[str, Any]]:
    pred = payload["pre_registered_predictions"]
    by_unit = {row["unit_id"]: row for row in rows}
    scope = {row["unit_id"]: row["machine_match"]
             for row in payload["t1"].get("scope_table") or []}
    table = [{
        "prediction": "card compiles at the unit-3 boundary",
        "expected": True,
        "observed": payload["t1"]["card"] is not None,
        "held": payload["t1"]["card"] is not None,
    }]
    for unit in pred["scope_matches"]:
        table.append({"prediction": "Scope matches %s" % unit,
                      "expected": True, "observed": scope.get(unit),
                      "held": scope.get(unit) is True})
    for unit in pred["scope_does_not_match"]:
        table.append({"prediction": "Scope does not match %s" % unit,
                      "expected": False, "observed": scope.get(unit),
                      "held": scope.get(unit) is False})
    for unit in pred["converts"]:
        row = by_unit.get(unit)
        observed = bool(row and row["applied_ops"]
                        and row["candidate_sources"]["supplied_probed"])
        table.append({"prediction": "%s converts" % unit, "expected": True,
                      "observed": observed, "held": observed})
    for unit in pred["does_not_convert"] + pred["abstains"]:
        row = by_unit.get(unit)
        observed = bool(row and not row["applied_ops"])
        table.append({"prediction": "%s does not deploy" % unit,
                      "expected": True, "observed": observed,
                      "held": observed})
    harm = sum(1 for row in rows if row["harm_event"])
    table.append({"prediction": "harm events = 0", "expected": 0,
                  "observed": harm, "held": harm == 0})
    facts = payload["verdict"]["facts"]
    table.append({
        "prediction": "A5 tail regret +0.7710 -> <= 0.20",
        "expected": "<= 0.20",
        "observed": round(facts["a5_tail_regret_l1"], 4),
        "held": facts["a5_tail_regret_l1"] <= 0.20})
    return table


def _l1_obligations() -> dict[str, Any]:
    return {
        "methods_package_unmodified": True,
        "try_tier_loo_untouched": True,
        "risk_tier_untouched": True,
        "execution_and_deployment_gates_untouched": True,
        "material_threshold_untouched": True,
        "prompt_model_budget_protocol_untouched": True,
        "sealed_artifacts_untouched": "Epilepsy2 and s1_oracle never enter "
                                      "any arm view",
        "no_new_units_operators_or_consumers": True,
        "producer_stage_not_rerun": True,
        "other_three_arms_are_v4_replay_grade_control": True,
        "guided_positive_counts_zero": True,
        "downloads": 0,
        "full_repo_pytest_not_run": True,
    }


def _l1_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = [
        "# L1 -- ladder revision v2 replay from the v4 boundary", "",
        "**Core positive effect moved: %s**" % (
            verdict.get("headline") or "not evaluated (offline gate only)"),
        "",
        "protocol: `%s`  git: `%s`  verdict: **%s**"
        % (payload["protocol_version"], payload["git_head"],
           verdict["verdict"]),
        "", verdict.get("reason", ""), "",
        "> %s" % payload["ladder_revision"], "",
        "## T1 offline gate", "",
        "| check | pass | evidence |", "|---|---|---|",
    ]
    for row in payload["t1"]["checks"]:
        ev = row["evidence"]
        cite = (ev.get("file_line") or ev.get("predicate_file_line")
                or ev.get("reader_file_line") or ev.get("rule_file_line")
                or ev.get("dual_gate_enforced_at") or "")
        lines.append("| %s | **%s** | `%s` |" % (row["check"], row["pass"],
                                                 cite))
    scope_table = payload["t1"].get("scope_table") or []
    if scope_table:
        lines += ["", "### Scope match precheck", "",
                  "| # | unit | role | machine match | served in Fast view |",
                  "|---|---|---|---|---|"]
        for row in scope_table:
            lines.append("| %s | `%s` | %s | **%s** | %s |" % (
                row["position"], row["unit_id"], row["role"],
                row["machine_match"], row["served_in_fast_view"]))
    boundary = payload.get("boundary_resume")
    if boundary:
        lines += ["", "## Boundary resume", "",
                  "- kind: **%s**" % boundary["kind"],
                  "- carried: %s" % boundary["carried"],
                  "- not carried: %s" % boundary["not_carried"],
                  "- producer stage re-run: %s"
                  % (not boundary["producer_stage_not_rerun"])]
    if payload.get("rows"):
        lines += ["", "## A5 tail, per unit", "",
                  "| # | role | unit | supplied in pool | self-proposed | "
                  "supplied probed | deployed | held-out | regret | worst "
                  "class | probes | LLM | fits |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in payload["rows"]:
            src = row["candidate_sources"]
            lines.append("| %s | %s | %s | %d | %d | %d | `%s` | %+.4f | "
                         "%+.4f | %+.4f | %d | %d | %d |" % (
                             row["position"], row["role"],
                             row["unit_id"].split("__")[0],
                             src["supplied_in_pool"],
                             src["self_proposed_in_pool"],
                             src["supplied_probed"],
                             ",".join(row["applied_ops"]) or "identity",
                             row["heldout_utility"], row["regret"],
                             row["worst_class_delta"], row["probes"],
                             row["llm_calls"], row["consumer_fits"]))
    if payload.get("prediction_table"):
        lines += ["", "## Pre-registered predictions", "",
                  "| prediction | expected | observed | held |",
                  "|---|---|---|---|"]
        for row in payload["prediction_table"]:
            lines.append("| %s | %s | %s | **%s** |" % (
                row["prediction"], row["expected"], row["observed"],
                row["held"]))
    control = payload.get("v4_control") or {}
    if control:
        lines += ["", "## v4 replay-grade control (tail units)", "",
                  "- %s" % control.get("note", ""), "",
                  "| arm | units | cumulative regret | harm | probes | LLM | "
                  "fits |", "|---|---|---|---|---|---|---|"]
        for arm in (ARM_STATIC, ARM_A3, ARM_K0, ARM_A5):
            row = control.get(arm)
            if not row:
                continue
            lines.append("| %s | %d | %+.4f | %d | %d | %d | %d |" % (
                arm, row["units"], row["cumulative_regret"],
                row["harm_events"], row["probes"], row["llm"],
                row["consumer_fits"]))
    ledger = payload.get("ledger") or {}
    lines += ["", "## Cost", "",
              "- LLM: %s / %s" % (ledger.get("llm"), ledger.get("llm_cap")),
              "- fits: %s / %s" % (ledger.get("fit"), ledger.get("fit_cap")),
              "- wall: %s s" % ledger.get("wall_seconds"),
              "- downloads: 0", "", "## Obligations", ""]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="S1-v2 forward course")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--freeze-r2", action="store_true")
    parser.add_argument("--freeze-v3", action="store_true")
    parser.add_argument("--freeze-v4", action="store_true")
    parser.add_argument("--l1-t1", action="store_true",
                        help="L1 offline gate only (0 LLM)")
    parser.add_argument("--l1-replay", action="store_true",
                        help="L1: T1 then the live A5 tail replay")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize", action="store_true",
                        help="re-render a run's artifact from its checkpoint")
    parser.add_argument("--seed", choices=sorted(SEEDS), default="r1")
    args = parser.parse_args()
    if args.freeze:
        return freeze()
    if args.freeze_r2:
        return freeze_r2()
    if args.freeze_v3:
        return freeze_v3()
    if args.freeze_v4:
        return freeze_v4()
    if args.l1_t1:
        return l1_replay(args.seed, t1_only=True)
    if args.l1_replay:
        return l1_replay(args.seed, resume=bool(args.resume))
    # The live entries are gated on Part 0.  The gate is read off the frozen
    # artifact rather than recomputed, so a run can never start on a course
    # the freeze refused.
    if not FREEZE_V4_JSON.is_file():
        parser.error("run --freeze-v4 first")
    frozen = json.loads(FREEZE_V4_JSON.read_text(encoding="utf-8"))
    verdict = (frozen.get("verdict") or {}).get("verdict")
    if verdict != "S1V2_COURSE_FROZEN_V4":
        print(json.dumps({
            "verdict": verdict,
            "reason": (frozen.get("verdict") or {}).get("reason"),
            "first_unmet_condition": frozen.get("first_unmet_condition"),
            "llm_spent": 0, "consumer_fits_spent": 0,
        }, ensure_ascii=False, indent=1))
        return 1
    return run_course(args.seed, resume=bool(args.resume),
                      finalize=bool(args.finalize))


if __name__ == "__main__":
    raise SystemExit(main())
