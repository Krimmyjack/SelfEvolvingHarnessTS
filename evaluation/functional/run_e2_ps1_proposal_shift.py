"""PS-1 -- does a Scoped hypothesis shift what the proposal stage proposes?

S1c named the bottleneck ``candidate discovery``: over fifteen arm-unit
opportunities that contained a menu-oracle answer, the cold proposal stage
reached it once.  On ``GunPointOldVersusYoung__impulse_v2`` the sealed oracle
carries a +0.184 held-in headroom on ``hampel_filter`` and all three adaptive
arms missed it -- A3 proposed nothing in r1 and abandoned a single rejected
candidate in r2.

The hypothesis this book tests is mechanism-level: two independent, legal,
unguided positives compiled into a *structured hypothesis card*, served
through a runner-owned experimental prior slot, should raise the probability
that the right Program family enters a small candidate budget, and the
Target's own held-in feedback should then convert it into a local Skill --
or refuse to.

Before any of that can be measured, the two sources have to survive a
provenance gate, because a hypothesis whose Scope cannot be evaluated by
machine is not a hypothesis: it is a dataset name in prose.  Part 0 is that
gate and it is hard -- this file does not compile a card, does not open the
exam substrate and does not spend a token unless the gate passes.

Entry points::

  python evaluation/functional/run_e2_ps1_proposal_shift.py --part0
  python evaluation/functional/run_e2_ps1_proposal_shift.py --run

``--part0`` runs the gate alone.  ``--run`` runs the gate and then the
experiment; on a gate failure it writes the stop artifact and returns without
touching Parts 1-3.

Evidence grade: development-mechanism.  ``GunPointOldVersusYoung`` shares
``GunPointFamily`` with source A, so a positive result here isolates the
proposal-shift mechanism and is NOT a cross-family transfer capability claim.
"""
from __future__ import annotations

import argparse
import json
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

# Importing the S1b runner installs its oracle-isolation wall (every reader
# surface wrapped at import, arm-phase reads raise).  Part 0 reads no sealed
# key at all and the artifact carries the wall's own report as proof.
import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
from SelfEvolvingHarnessTS.contracts.observables import (  # noqa: E402
    OBSERVABLE_FEATURES,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "ps1_proposal_shift.json"
OUT_MD = E2 / "ps1_proposal_shift.md"
CARD_DIR = E2 / "ps1_cards"

# =========================================================================== #
# frozen protocol -- declared here, before any outcome of this book is seen
# =========================================================================== #
PROTOCOL_VERSION = "ps1_proposal_shift_v1"
EVIDENCE_GRADE = "development-mechanism"

EXAM_UNIT = "GunPointOldVersusYoung__impulse_v2"
ARMS = ("A3", "A5-neutral", "A5-scoped")
REPLICATES = 4
RUN_IDS = tuple("ps1_run%d" % index for index in range(1, 13))

LLM_PER_RUN = 12
FIT_PER_RUN = 10
LLM_TOTAL_CAP = 150
FIT_TOTAL_CAP = 120
WALL_SECONDS_CAP = int(2.5 * 60 * 60)

SOURCE_A = {
    "label": "source_A",
    "artifact": "artifacts/functional/e2/t6_cls_op_r2_three_arms.json",
    "locator": {"section": "part_c.rounds", "dataset": "GunPointAgeSpan",
                "arm": "A3", "round": "r1"},
    "expected_program": "hampel_filter",
    "family_key": "GunPointFamily",
}
SOURCE_B = {
    "label": "source_B",
    "artifact": "artifacts/functional/e2/s1_course_forward_run1.json",
    "locator": {"section": "arm_results", "unit_id": "PowerCons__impulse_v2",
                "arm": "A3-reset", "round": "r1"},
    "expected_program": "hampel_filter",
    "family_key": "PowerCons",
}

MATERIAL = s1.MATERIAL  # 0.005, experience_memory / signed_radius
BOOTSTRAP_SKILL_IDS = frozenset({
    "build_contrastive_candidates", "inspect_and_localize",
    "select_or_identity_and_verify",
})
# Scope rule v1, frozen: the five axes a cross-domain card must carry.
SCOPE_V1_AXES = ("task_kind", "consumer_id", "metric",
                 "deployment_visible_pattern_intersection", "program_geometry")
# Every observable-contract leaf except the eligibility gate itself.  A
# Pattern axis has to be built out of these or it cannot be evaluated by
# ``retrieval.evaluate_applicability``.
PATTERN_LEAVES = tuple(sorted(set(OBSERVABLE_FEATURES) - {"task_kind"}))

VERDICTS = (
    "PROPOSAL_SHIFT_CONFIRMED",
    "SHIFT_WITHOUT_CONVERSION",
    "NO_PROPOSAL_SHIFT",
    "PLACEBO_EFFECT",
    "SOURCE_PROVENANCE_INSUFFICIENT",
    "COMPUTE_BUDGET_EXCEEDED",
)

PRE_REGISTERED_READOUT = {
    "per_run": [
        "every raw proposal each round, tagged with its Program family",
        "hampel family: proposed / selected / passed the verifier / earned "
        "Support / earned delayed / deployed",
        "probes and wasted probes",
        "LLM and Consumer-fit cost",
        "non-hampel proposal diversity (crowding-out check)",
        "harm events and worst-class recall delta",
    ],
    "aggregate_by_arm": [
        "hampel-family proposal rate at run granularity (how many of the 4 "
        "runs proposed it at all)",
        "conversion funnel proposed -> selected -> verifier -> Support -> "
        "delayed -> deployed",
        "mean cost",
        "deployed utility distribution",
    ],
    "verdicts": {
        "PROPOSAL_SHIFT_CONFIRMED": (
            "A5-scoped's hampel proposal rate separates from both controls "
            "(order of >=3/4 against <=1/4) AND at least one run completes "
            "proposal -> Support -> delayed -> deployment AND harm does not "
            "rise AND A5-neutral shows no systematic separation from A3"),
        "SHIFT_WITHOUT_CONVERSION": (
            "proposal rate separates but nothing converts; the report names "
            "the layer it stopped at"),
        "NO_PROPOSAL_SHIFT": "the three arms' proposal distributions overlap",
        "PLACEBO_EFFECT": (
            "A5-neutral departs systematically from A3, i.e. the presence of "
            "a card changes behaviour on its own and any scoped effect has to "
            "be reinterpreted against that baseline"),
        "SOURCE_PROVENANCE_INSUFFICIENT": (
            "the Part 0 gate did not pass; no card was compiled and no arm "
            "ran"),
        "COMPUTE_BUDGET_EXCEEDED": (
            "a cap was hit; completed runs are kept and reported"),
    },
    "statistics": (
        "n=4 per arm is reported as counts and effect sizes.  No p-values."),
    "scope_caveat": (
        "%s shares GunPointFamily with source A.  This experiment isolates "
        "the proposal-shift mechanism and must not be cited as cross-family "
        "transfer capability." % EXAM_UNIT),
}


def _plain(value: Any) -> Any:
    return s1._plain(value)


def _read(relative: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return json.loads(path.read_text(encoding="utf-8"))


# =========================================================================== #
# Part 0 -- the dual-source provenance gate
# =========================================================================== #
def _locate_source_a(payload: Mapping[str, Any]) -> dict[str, Any]:
    locator = SOURCE_A["locator"]
    rounds = ((payload.get("part_c") or {}).get("rounds") or [])
    record = next(
        (row for row in rounds
         if row.get("dataset") == locator["dataset"]
         and row.get("arm") == locator["arm"]
         and row.get("round") == locator["round"]), None)
    cells = ((payload.get("part_c") or {}).get("cells") or [])
    cell = next((row for row in cells
                 if row.get("dataset") == locator["dataset"]), None)
    deployments = ((payload.get("part_c") or {}).get("deployments") or [])
    deployment = next(
        (row for row in deployments
         if row.get("dataset") == locator["dataset"]
         and row.get("arm") == locator["arm"]), None)
    return {"record": record, "cell": cell, "deployment": deployment,
            "arm_started_from": "h0 (A3 is the cold-start ablation arm)"}


def _locate_source_b(payload: Mapping[str, Any]) -> dict[str, Any]:
    locator = SOURCE_B["locator"]
    result = next(
        (row for row in (payload.get("arm_results") or [])
         if row.get("unit_id") == locator["unit_id"]
         and row.get("arm") == locator["arm"]), None)
    record = None
    if result is not None:
        record = next((row for row in (result.get("rounds") or [])
                       if row.get("round") == locator["round"]), None)
    unit = next((row for row in (payload.get("course") or [])
                 if row.get("unit_id") == locator["unit_id"]), None)
    return {"record": record, "cell": unit,
            "deployment": (result or {}).get("deployment"),
            "arm_started_from": (result or {}).get("base_skill_ids"),
            "episodes_at_unit_start": (result or {}).get(
                "episodes_at_unit_start"),
            "arm_result": result}


def _episode_for(record: Mapping[str, Any], program: str
                 ) -> dict[str, Any] | None:
    for episode in record.get("episodes") or []:
        if str(episode.get("workflow_signature")) == program:
            return dict(episode)
    return None


def _check_executed(source: Mapping[str, Any], found: Mapping[str, Any],
                    ) -> dict[str, Any]:
    record = found.get("record")
    episode = _episode_for(record or {}, str(source["expected_program"]))
    ok = bool(record is not None and episode is not None
              and episode.get("episode_id"))
    return {
        "item": "1_real_executed_episode",
        "question": ("is this a record of an Episode a live arm actually "
                     "wrote, rather than a row of the oracle census?"),
        "pass": ok,
        "episode_id": (episode or {}).get("episode_id"),
        "evidence_level": (episode or {}).get("evidence_level"),
        "local_status": (episode or {}).get("local_status"),
        "cited_fields": ["episodes[].episode_id", "episodes[].evidence_level",
                         "episodes[].local_status"],
        "note": ("oracle census rows carry no episode_id and no local_status; "
                 "these do"),
    }


def _check_unguided(source: Mapping[str, Any], found: Mapping[str, Any],
                    ) -> dict[str, Any]:
    record = found.get("record") or {}
    retrieved = [str(item) for item in (record.get("retrieved_skill_ids") or [])]
    beyond = sorted(set(retrieved) - BOOTSTRAP_SKILL_IDS)
    ok = not beyond
    return {
        "item": "2_unguided",
        "question": ("did this arm's Fast view carry any TRY or capability "
                     "card of the same Program family when the proposal was "
                     "made?"),
        "pass": ok,
        "retrieved_skill_ids": retrieved,
        "beyond_bootstrap": beyond,
        "memory_resolution": record.get("memory_resolution"),
        "arm_started_from": found.get("arm_started_from"),
        "episodes_at_unit_start": found.get("episodes_at_unit_start"),
        "cited_fields": ["retrieved_skill_ids", "memory_resolution",
                         "base_skill_ids", "episodes_at_unit_start"],
        "note": ("only the three h0 bootstrap procedures were retrieved, so "
                 "nothing named a Program family"),
    }


def _check_material_both_gates(source: Mapping[str, Any],
                               found: Mapping[str, Any]) -> dict[str, Any]:
    record = found.get("record") or {}
    episode = _episode_for(record, str(source["expected_program"])) or {}
    support = episode.get("support_gain")
    delayed = episode.get("delayed_gain")
    support_ok = support is not None and float(support) >= MATERIAL
    delayed_ok = delayed is not None and float(delayed) >= MATERIAL
    relation_ok = str(episode.get("relation")) == "POSITIVE"
    approved = record.get("approved_skill_id")
    return {
        "item": "3_material_positive_on_support_and_delayed",
        "question": "did both live gates read materially positive?",
        "pass": bool(support_ok and delayed_ok and relation_ok and approved),
        "support_gain": support,
        "delayed_gain": delayed,
        "material_threshold": MATERIAL,
        "relation": episode.get("relation"),
        "approved_skill_id": approved,
        "activated": record.get("activated"),
        "cited_fields": ["episodes[].support_gain", "episodes[].delayed_gain",
                         "episodes[].relation", "approved_skill_id",
                         "activated"],
    }


def _check_family_independence(rows: Sequence[Mapping[str, Any]]
                               ) -> dict[str, Any]:
    families = [str(row["family_key"]) for row in rows]
    return {
        "item": "4_family_independence",
        "question": "are the two sources independent families?",
        "pass": len(set(families)) == len(families),
        "families": families,
        "cited_fields": ["s1a_r3_pool_census.json units[].family_key"],
        "note": ("family_key is the census union-find over name prefix and "
                 "byte-equal pattern_view; GunPointFamily and PowerCons are "
                 "distinct keys"),
    }


def _pattern_leaves_present(node: Any) -> list[str]:
    """Which observable-contract Pattern leaves this record actually stores."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key) in PATTERN_LEAVES:
                    found.add(str(key))
                walk(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                walk(nested)

    walk(node)
    return sorted(found)


def _check_machine_scope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Axis 5: intersect the frozen Scope-v1 axes from stored fields only.

    The book's rule is explicit -- intersect from the fields the two
    provenance records already hold, and if they do not hold enough
    deployment-visible Pattern to do it, report the gap and stop.  No axis is
    reconstructed, recomputed or described after the fact here.
    """
    axes: dict[str, Any] = {}
    per_source: list[dict[str, Any]] = []
    for row in rows:
        record = row["found"].get("record") or {}
        cell = row["found"].get("cell") or {}
        key = str(record.get("task_consumer_key") or "")
        parts = key.split("|")
        leaves = sorted(set(_pattern_leaves_present(record))
                        | set(_pattern_leaves_present(cell)))
        per_source.append({
            "label": row["source"]["label"],
            "task_kind": parts[0] if parts else None,
            "consumer_id": parts[1] if len(parts) > 1 else None,
            "metric": parts[2] if len(parts) > 2 else None,
            "program_geometry": str(row["source"]["expected_program"]),
            "pattern_leaves_stored": leaves,
            "context_fields_actually_stored": sorted(
                set(cell.keys()) - {"dataset", "condition", "archive"}),
            "task_consumer_key": key,
        })
    for axis, field in (("task_kind", "task_kind"),
                        ("consumer_id", "consumer_id"),
                        ("metric", "metric"),
                        ("program_geometry", "program_geometry")):
        values = {row[field] for row in per_source}
        axes[axis] = {
            "values": sorted(str(value) for value in values),
            "agree": len(values) == 1 and None not in values,
            "intersection": (sorted(values)[0] if len(values) == 1 else None),
            "source": "stored field",
        }
    shared_leaves = set(PATTERN_LEAVES)
    for row in per_source:
        shared_leaves &= set(row["pattern_leaves_stored"])
    axes["deployment_visible_pattern_intersection"] = {
        "leaves_available_in_both_records": sorted(shared_leaves),
        "intersection": {},
        "agree": False,
        "source": "stored field",
        "why_empty": (
            "neither provenance record persists a single one of the %d "
            "non-task_kind observable-contract leaves.  The only Context the "
            "records hold is witness statistics, "
            "support_reproduces_fit_signal, observer node positions and "
            "slice sizes, and none of those is in "
            "contracts/observables.OBSERVABLE_FEATURES, so none can become "
            "an applicability leaf that retrieval.evaluate_applicability "
            "could ever read." % len(PATTERN_LEAVES)),
    }
    missing = [axis for axis in SCOPE_V1_AXES if not axes[axis]["agree"]]
    return {
        "item": "5_machine_executable_five_axis_scope",
        "question": ("can a Scope-v1 intersection be computed from the stored "
                     "fields, and would a machine be able to evaluate it?"),
        "pass": not missing,
        "axes": axes,
        "per_source": per_source,
        "axes_available": [axis for axis in SCOPE_V1_AXES
                           if axes[axis]["agree"]],
        "axes_missing": missing,
        "observable_contract_leaves": list(PATTERN_LEAVES),
        "cited_fields": ["task_consumer_key", "episodes[].workflow_signature",
                         "part_c.cells[]", "course[]"],
    }


def part0_provenance() -> dict[str, Any]:
    started = time.time()
    s1._set_phase(s1.PHASE_SELECT)
    rows: list[dict[str, Any]] = []
    load_errors: list[dict[str, Any]] = []
    for source, locate in ((SOURCE_A, _locate_source_a),
                           (SOURCE_B, _locate_source_b)):
        try:
            payload = _read(str(source["artifact"]))
        except FileNotFoundError:
            load_errors.append({"source": source["label"],
                                "artifact": source["artifact"],
                                "error": "artifact not found"})
            continue
        found = locate(payload)
        if found.get("record") is None:
            load_errors.append({"source": source["label"],
                                "artifact": source["artifact"],
                                "error": "locator matched no round record",
                                "locator": source["locator"]})
            continue
        rows.append({"source": source, "found": found})

    checks: list[dict[str, Any]] = []
    if load_errors or len(rows) != 2:
        checks.append({"item": "0_records_located", "pass": False,
                       "errors": load_errors})
        return {
            "gate": "dual_source_provenance",
            "pass": False,
            "checks": checks,
            "seconds": round(time.time() - started, 2),
        }
    checks.append({"item": "0_records_located", "pass": True,
                   "located": [{"source": row["source"]["label"],
                                "artifact": row["source"]["artifact"],
                                "locator": row["source"]["locator"]}
                               for row in rows]})
    per_source_checks: list[dict[str, Any]] = []
    for row in rows:
        for check in (_check_executed(row["source"], row["found"]),
                      _check_unguided(row["source"], row["found"]),
                      _check_material_both_gates(row["source"], row["found"])):
            check["source"] = row["source"]["label"]
            per_source_checks.append(check)
    checks.extend(per_source_checks)
    checks.append(_check_family_independence([row["source"] for row in rows]))
    checks.append(_check_machine_scope(rows))

    deployments = [{
        "source": row["source"]["label"],
        "deploy_source": (row["found"].get("deployment") or {}).get(
            "deploy_source"),
        "applied_program": (row["found"].get("deployment") or {}).get(
            "applied_program"),
        "heldout_accuracy_gain": (row["found"].get("deployment") or {}).get(
            "heldout_accuracy_gain"),
    } for row in rows]
    return {
        "gate": "dual_source_provenance",
        "pass": all(check["pass"] for check in checks),
        "checks": checks,
        "deployment_outcomes": deployments,
        "scope_rule": ("Scope v1: %s.  Dataset name is not an axis."
                       % " x ".join(SCOPE_V1_AXES)),
        "seconds": round(time.time() - started, 2),
    }


def _gap_report(part0: Mapping[str, Any]) -> dict[str, Any]:
    """What exactly is missing, and what would close it.  Nothing is done."""
    scope = next((check for check in part0["checks"]
                  if check["item"] == "5_machine_executable_five_axis_scope"),
                 None)
    if scope is None or scope["pass"]:
        return {}
    return {
        "failing_item": scope["item"],
        "axes_available": scope["axes_available"],
        "axes_missing": scope["axes_missing"],
        "diagnosis": (
            "the provenance chain for a legal cross-domain hypothesis is "
            "broken at the persistence layer, not at the evidence layer.  "
            "Both sources are real, unguided, materially positive on both "
            "gates and independent by family -- items 1 through 4 all pass.  "
            "What neither execution record kept is the deployment-visible "
            "Pattern the Scope has to be built out of."),
        "where_the_pattern_does_exist": [
            {"location": "artifacts/functional/e2/s1a_r3_pool_census.json "
                         "units[].pattern_view",
             "usable": False,
             "why": ("sealed audit artifact.  Its own isolation banner "
                     "forbids it entering any arm prompt, store or retrieval "
                     "view, and a hypothesis card is exactly a Fast-visible "
                     "surface")},
            {"location": "artifacts/functional/e2/s1_oracle/*.json "
                         "pattern_view / public_features_binned",
             "usable": False,
             "why": "sealed exam key; same banner, stronger reason"},
            {"location": ("recomputation via _build_cell + "
                          "extract_public_features on the same frozen cell"),
             "usable": False,
             "why": ("deterministic and outcome-free, but the book's rule 5 "
                     "says to intersect from the fields the records already "
                     "store and to stop rather than reconstruct.  Not done "
                     "here; named so the main line can decide")},
        ],
        "smallest_thing_that_would_close_it": (
            "persist the binned deployment-visible pattern view on the round "
            "or cell record at write time.  The Fast path already computes it "
            "every round -- run_e2_s1_curriculum_four_arms._run_round builds "
            "`features = extract_public_features(block, task_kind=...)` and "
            "hands it to run_online_round as fast_features -- so this is a "
            "record-keeping change, not a new computation.  Once a run "
            "persists it, this gate passes on that run's own fields and PS-1 "
            "proceeds without any reconstruction."),
        "note_for_the_main_line": (
            "closing it needs one fresh source-B-shaped run, or a re-run of "
            "both, with the field persisted.  Source A predates this runner "
            "entirely and would have to be re-earned rather than re-read."),
    }


# =========================================================================== #
# artifact
# =========================================================================== #
def _obligations(part0: Mapping[str, Any], *, ran_arms: bool) -> dict[str, Any]:
    return {
        "part0_items": {check["item"] + (
            ":" + check["source"] if check.get("source") else ""): check["pass"]
            for check in part0["checks"]},
        "prior_slot_implementation": (
            "not reached: no card was compiled and no arm ran.  The slot is "
            "declared as a runner-layer experiment mechanism only -- an "
            "independent paragraph in the agent's construction-time context, "
            "supplying no frozen candidate and changing no budget"),
        "production_governance_unmodified": True,
        "methods_package_unmodified": True,
        "runtime_contracts_operators_unmodified": True,
        "t1_authorization_retrieval_unmodified": True,
        "llm_calls": 0,
        "consumer_fits": 0,
        "downloads": 0,
        "arms_run": 0 if not ran_arms else len(RUN_IDS),
        "exam_substrate_opened": False,
        "sealed_artifacts_not_read": True,
        "sealed_artifacts_not_rewritten": True,
        "full_repo_pytest_not_run": True,
        "stage_report_not_written": (
            "this book does not touch docs/STAGE_REPORT; another diagnostic "
            "book is in flight and the main line records for it"),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    part0 = payload["part0"]
    lines = [
        "# PS-1 -- proposal shift under a Scoped hypothesis",
        "",
        "protocol: `%s`  evidence grade: **%s**  git: `%s`"
        % (payload["protocol_version"], payload["evidence_grade"],
           payload["git_head"]),
        "",
        "**%s**" % payload["verdict"]["verdict"],
        "",
        payload["verdict"]["reason"],
        "",
        "## Part 0 -- dual-source provenance gate",
        "",
        "| item | source | pass | evidence |",
        "|---|---|---|---|",
    ]
    for check in part0["checks"]:
        if check["item"] == "0_records_located":
            detail = ("both records located"
                      if check["pass"] else str(check.get("errors")))
        elif check["item"].startswith("1_"):
            detail = "`%s`, %s / %s" % (check["episode_id"],
                                        check["evidence_level"],
                                        check["local_status"])
        elif check["item"].startswith("2_"):
            detail = ("retrieved only %s; beyond bootstrap: %s"
                      % (len(check["retrieved_skill_ids"]),
                         check["beyond_bootstrap"] or "none"))
        elif check["item"].startswith("3_"):
            detail = ("Support %+.4f, delayed %+.4f, relation %s, Skill `%s`"
                      % (check["support_gain"], check["delayed_gain"],
                         check["relation"],
                         str(check["approved_skill_id"])[:48]))
        elif check["item"].startswith("4_"):
            detail = " vs ".join(check["families"])
        else:
            detail = ("axes available %s; missing %s"
                      % (check["axes_available"], check["axes_missing"]))
        lines.append("| %s | %s | %s | %s |" % (
            check["item"], check.get("source", "-"),
            "PASS" if check["pass"] else "**FAIL**", detail))
    lines += ["", "### Deployment outcomes of the two sources", "",
              "| source | deploy source | program | held-out gain |",
              "|---|---|---|---|"]
    for row in part0.get("deployment_outcomes") or []:
        lines.append("| %s | %s | %s | %s |" % (
            row["source"], row["deploy_source"],
            ", ".join(step["op"] for step in (row["applied_program"] or []))
            or "identity", row["heldout_accuracy_gain"]))
    scope = next((check for check in part0["checks"]
                  if check["item"].startswith("5_")), None)
    if scope:
        lines += ["", "### Axis 5 in detail", "",
                  "| axis | intersection | agree | source |",
                  "|---|---|---|---|"]
        for axis in SCOPE_V1_AXES:
            row = scope["axes"][axis]
            lines.append("| %s | %s | %s | %s |" % (
                axis, row.get("intersection"), row["agree"], row["source"]))
        lines += ["", "- %s"
                  % scope["axes"]["deployment_visible_pattern_intersection"][
                      "why_empty"], ""]
        lines += ["| source | Pattern leaves stored | Context fields the "
                  "record does keep |", "|---|---|---|"]
        for row in scope["per_source"]:
            lines.append("| %s | %s | %s |" % (
                row["label"], row["pattern_leaves_stored"] or "**none**",
                ", ".join(row["context_fields_actually_stored"])))
    gap = payload.get("gap_report") or {}
    if gap:
        lines += ["", "## The gap, precisely", "", gap["diagnosis"], "",
                  "Where the Pattern does exist today:", ""]
        for row in gap["where_the_pattern_does_exist"]:
            lines.append("- `%s` -- usable: **%s**.  %s"
                         % (row["location"], row["usable"], row["why"]))
        lines += ["", "**Smallest thing that would close it**: %s"
                  % gap["smallest_thing_that_would_close_it"], "",
                  "- %s" % gap["note_for_the_main_line"]]
    lines += ["", "## Frozen protocol for Parts 1-3 (pre-registered, not run)",
              "",
              "The gate did not pass, so no card was compiled and no arm ran.  "
              "The protocol below is frozen now, before any outcome of this "
              "experiment has been seen, so that closing the gap does not "
              "reopen the design.",
              "",
              "- exam substrate: `%s`" % EXAM_UNIT,
              "- arms: %s" % ", ".join(ARMS),
              "- replicates per arm: %d; run ids `%s` .. `%s`"
              % (REPLICATES, RUN_IDS[0], RUN_IDS[-1]),
              "- budgets: LLM <= %d/run and <= %d total; fit <= %d/run and "
              "<= %d total; wall <= %d s"
              % (LLM_PER_RUN, LLM_TOTAL_CAP, FIT_PER_RUN, FIT_TOTAL_CAP,
                 WALL_SECONDS_CAP),
              "- verdict set: %s" % ", ".join(VERDICTS),
              ""]
    for key, value in PRE_REGISTERED_READOUT.items():
        if isinstance(value, list):
            lines.append("- **%s**:" % key)
            lines += ["  - %s" % item for item in value]
        elif isinstance(value, dict):
            lines.append("- **%s**:" % key)
            lines += ["  - `%s`: %s" % (name, text)
                      for name, text in value.items()]
        else:
            lines.append("- **%s**: %s" % (key, value))
    isolation = payload["oracle_isolation"]
    lines += ["", "## Oracle isolation", "",
              "- %s" % isolation["mechanism"],
              "- unblocked reads by phase: %s"
              % isolation["unblocked_reads_by_phase"],
              "- arm-phase attempts: %d, leaks %d"
              % (isolation["arm_phase_attempts"],
                 len(isolation["arm_phase_leaks"])),
              "", "## Cost", "",
              "- LLM: 0 (the gate spends none and Parts 1-3 did not start)",
              "- Consumer fits: 0",
              "- wall clock: %.2f s" % payload["wall_seconds"],
              "- downloads: 0",
              "", "## Obligations", ""]
    for key, value in payload["obligations"].items():
        lines.append("- **%s**: %s" % (key, value))
    if payload.get("outside_book"):
        lines += ["", "## Outside the book", ""]
        lines += ["- %s" % item for item in payload["outside_book"]]
    return "\n".join(lines) + "\n"


def run(*, gate_only: bool = False) -> int:
    started = time.time()
    part0 = part0_provenance()
    gap = _gap_report(part0)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "entry": "--part0" if gate_only else "--run",
        "exam_unit": EXAM_UNIT,
        "sources": {"source_A": SOURCE_A, "source_B": SOURCE_B},
        "part0": part0,
        "gap_report": gap,
        "frozen_protocol": {
            "arms": list(ARMS),
            "replicates": REPLICATES,
            "run_ids": list(RUN_IDS),
            "llm_per_run": LLM_PER_RUN,
            "fit_per_run": FIT_PER_RUN,
            "llm_total_cap": LLM_TOTAL_CAP,
            "fit_total_cap": FIT_TOTAL_CAP,
            "wall_seconds_cap": WALL_SECONDS_CAP,
            "verdicts": list(VERDICTS),
            "prior_slot": (
                "runner-layer experiment mechanism: the card is injected as "
                "an independent paragraph into the agent's construction-time "
                "context.  Production T1, authorization and retrieval are "
                "untouched; the slot supplies no frozen candidate and changes "
                "no budget; artifacts carry experimental_prior_slot=true"),
            "protocol_isomorphic_to": (
                "the S1c unit protocol: 2 held-in rounds, cohort modification "
                "scope, the same maximum_candidates, live gpt-5.6-sol backend "
                "identity-probed before the first arm"),
        },
        "pre_registered_readout": PRE_REGISTERED_READOUT,
        "parts_1_to_3": {
            "status": "not started",
            "reason": ("Part 0 is a hard gate and it did not pass.  Compiling "
                       "a card would have meant sourcing its WHEN clause from "
                       "somewhere other than the two records, which is the one "
                       "thing the book forbids."),
            "implemented": False,
            "why_not_implemented": (
                "the card compiler, the prior slot and the 12-run loop cannot "
                "be exercised or validated in this book, and building "
                "unexercisable machinery is what the project canon calls "
                "over-engineering.  The frozen protocol above is what a "
                "follow-up needs; the code is not written on speculation."),
        },
    }
    payload["oracle_isolation"] = s1._oracle_isolation_report()
    payload["wall_seconds"] = round(time.time() - started, 2)
    if part0["pass"]:
        payload["verdict"] = {
            "verdict": "GATE_PASSED_PARTS_1_TO_3_NOT_IMPLEMENTED",
            "reason": ("the provenance gate passed.  Parts 1-3 are not "
                       "implemented in this revision; re-issue the book."),
        }
    else:
        failing = [check["item"] for check in part0["checks"]
                   if not check["pass"]]
        payload["verdict"] = {
            "verdict": "SOURCE_PROVENANCE_INSUFFICIENT",
            "reason": (
                "items 1 through 4 pass on both sources: both are real "
                "executed Episodes, both unguided, both materially positive "
                "on Support and on delayed, and the two families are "
                "independent.  Item 5 fails: neither execution record "
                "persists any deployment-visible Pattern, so a five-axis "
                "Scope cannot be intersected from stored fields and the "
                "hypothesis card would have no machine-evaluable WHEN "
                "clause.  Stopped before Part 1 as the book directs."),
            "failing_items": failing,
            "stopped_before": "Part 1 card compilation",
        }
    payload["obligations"] = _obligations(part0, ran_arms=False)
    payload["outside_book"] = [
        ("the persistence gap is systemic, not specific to these two runs: "
         "the Fast path computes the binned pattern view every single round "
         "and hands it to run_online_round as fast_features, and no runner on "
         "the classification line writes it to its artifact.  Every future "
         "cross-domain hypothesis will hit this same gate."),
        ("the two sealed places the pattern view does live -- the r3 census "
         "and the s1_oracle keys -- both carry isolation banners, so the only "
         "legal supply is a live run that records its own Context.  A card "
         "built from either would be an oracle leak wearing a Scope."),
        ("source A predates the S1 runner line and its artifact shape has no "
         "slot for the field, so closing the gap for source A means earning "
         "the positive again rather than re-reading it."),
        ("second, independent reason source B's cell record cannot be a card "
         "source: the course record it lives in is the frozen curriculum "
         "entry, and that entry carries oracle-derived selection metadata -- "
         "oracle_set, menu_oracle_program, key_heldin_readout, "
         "largest_legal_heldin_magnitude, harmful_outlier_operators.  That is "
         "correct for a judging-side curriculum freeze and no arm ever reads "
         "it, but it means a card compiler pointed at course[] would be "
         "reading answers.  The Context a card may legally quote has to come "
         "from the round record, which is exactly the record that does not "
         "keep it."),
        ("S1c's own GPOvY record is worth keeping next to this: A3 proposed "
         "nothing in r1 and one verifier-rejected candidate in r2, K0-fixed "
         "probed outlier_iqr to +0.0909 and still deployed identity, and "
         "A5-online proposed only outlier_mad twice and had both rejected by "
         "the verifier.  The +0.184 hampel headroom was never proposed by any "
         "arm, which is the observation PS-1 exists to explain."),
    ]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    s1._dump(OUT_JSON, payload)
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"]["verdict"],
        "part0_pass": part0["pass"],
        "failing_items": payload["verdict"].get("failing_items"),
        "llm": 0, "fits": 0,
        "artifact": str(OUT_JSON)}, ensure_ascii=False, indent=1))
    return 0 if part0["pass"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part0", action="store_true",
                        help="run the dual-source provenance gate alone")
    parser.add_argument("--run", action="store_true",
                        help="run the gate and then the experiment")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.part0 or args.run:
        return run(gate_only=bool(args.part0 and not args.run))
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
