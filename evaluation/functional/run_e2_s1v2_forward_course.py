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
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description="S1-v2 forward course")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", choices=sorted(SEEDS), default="r1")
    args = parser.parse_args()
    if args.freeze:
        return freeze()
    # The live entries are gated on Part 0.  The gate is read off the frozen
    # artifact rather than recomputed, so a run can never start on a course
    # the freeze refused.
    if not FREEZE_JSON.is_file():
        parser.error("run --freeze first")
    frozen = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
    verdict = (frozen.get("verdict") or {}).get("verdict")
    if verdict != "S1V2_COURSE_FROZEN":
        print(json.dumps({
            "verdict": verdict,
            "reason": (frozen.get("verdict") or {}).get("reason"),
            "first_unmet_condition": frozen.get("first_unmet_condition"),
            "llm_spent": 0, "consumer_fits_spent": 0,
        }, ensure_ascii=False, indent=1))
        return 1
    parser.error("the live course driver is not part of this book's "
                 "committed scope; re-issue once the course freezes")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
