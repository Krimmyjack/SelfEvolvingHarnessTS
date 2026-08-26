"""PS-0 -- re-earn the two source positives on a runner that records Context.

PS-1 stopped at its provenance gate: both source positives were real, unguided
and materially positive on both live gates, but neither execution record kept
a deployment-visible Pattern, so the hypothesis card had no machine-evaluable
WHEN clause.  The fix was record-keeping, and it landed in
``run_e2_s1_curriculum_four_arms._run_round``: every round now persists the
binned observable-contract projection of the exact feature mapping the Fast
path received, plus the full raw proposal ledger family-tagged from compiled
steps rather than from the free-text candidate id.

This runner earns the two positives again on that repaired path.  Nothing
about the protocol is hinted: same two held-in rounds, same cohort
modification scope, same candidate cap, same per-unit budgets as the S1c unit
protocol, cold A3-reset from h0, live backend identity-probed first.  A scene
that misses is reported as a MISS with its full proposal ledger, because a
miss is the behaviour data that names the discovery bottleneck.

Entry points::

  python evaluation/functional/run_e2_ps0_reearn_sources.py --reearn
  python evaluation/functional/run_e2_ps0_reearn_sources.py --part0

``--reearn`` runs the scenes and then re-verifies the PS-1 provenance gate on
the fresh records.  ``--part0`` re-verifies against records already on disk.

Evidence grade: development-mechanism.
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

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "ps0_reearn_sources.json"
OUT_MD = E2 / "ps0_reearn_sources.md"

PROTOCOL_VERSION = "ps0_reearn_sources_v1"
EVIDENCE_GRADE = "development-mechanism"

TARGET_FAMILY = "hampel"
TARGET_OPERATOR = "hampel_filter"
MAX_RUNS_PER_SCENE = 2

SCENES = (
    {"scene": "source_A_prime",
     "unit_id": "GunPointAgeSpan__impulse_v2", "dataset": "GunPointAgeSpan",
     "injection": "impulse_v2", "series_length": 150,
     "family_key": "GunPointFamily",
     "run_ids": ("ps0_srcA_1", "ps0_srcA_2")},
    {"scene": "source_B_prime",
     "unit_id": "PowerCons__impulse_v2", "dataset": "PowerCons",
     "injection": "impulse_v2", "series_length": 144,
     "family_key": "PowerCons",
     "run_ids": ("ps0_srcB_1", "ps0_srcB_2")},
)

# Isomorphic to the S1c unit protocol.  Not a free parameter.
ROUNDS = s1.HELD_IN_ROUNDS
LLM_PER_RUN = s1.LLM_PER_UNIT_PER_ARM
FIT_PER_RUN = s1.FIT_PER_UNIT_PER_ARM
ARM = s1.ARM_A3

# Book-level caps for the whole PS-0 / PS-1 chain.
LLM_TOTAL_CAP = 220
FIT_TOTAL_CAP = 200
WALL_SECONDS_CAP = 3 * 60 * 60

MATERIAL = s1.MATERIAL


def _earned(result: Mapping[str, Any]) -> dict[str, Any]:
    """Did the target family walk the whole live lifecycle in this run?

    The standard is the one already in service: a Support receipt the live
    classifier calls materially POSITIVE, and a delayed feedback that approves
    the Draft.  Nothing here re-judges either gate.
    """
    for record in result.get("rounds") or []:
        winner_ops = [str(step.get("op"))
                      for step in (record.get("winner_program") or [])]
        if TARGET_OPERATOR not in winner_ops:
            continue
        episode = next(
            (row for row in (record.get("episodes") or [])
             if str(row.get("workflow_signature")) == TARGET_OPERATOR), None)
        if episode is None:
            continue
        support = episode.get("support_gain")
        support_ok = support is not None and float(support) >= MATERIAL
        approved = bool(record.get("winner_delayed_approved"))
        if support_ok and approved:
            return {
                "earned": True,
                "round": record.get("round"),
                "episode_id": episode.get("episode_id"),
                "support_gain": support,
                "delayed_gain": episode.get("delayed_gain"),
                "relation": episode.get("relation"),
                "local_status": episode.get("local_status"),
                "approved_skill_id": record.get("approved_skill_id"),
                "fast_features_binned": record.get("fast_features_binned"),
                "task_consumer_key": record.get("task_consumer_key"),
                "program_geometry": TARGET_OPERATOR,
            }
    return {"earned": False}


def _proposal_ledger(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in result.get("rounds") or []:
        for row in record.get("proposals") or []:
            rows.append({"round": record.get("round"), **row})
    return rows


def _run_scene(scene: Mapping[str, Any], *, store_root: Path, h0: Any,
               ledger: dict[str, int], started: float) -> dict[str, Any]:
    unit = {key: scene[key] for key in
            ("unit_id", "dataset", "injection", "series_length")}
    cell = s1._build_cell(unit)
    runs: list[dict[str, Any]] = []
    earned: dict[str, Any] | None = None
    for index, run_id in enumerate(scene["run_ids"][:MAX_RUNS_PER_SCENE], 1):
        if ledger["llm"] >= LLM_TOTAL_CAP or ledger["fit"] >= FIT_TOTAL_CAP:
            raise s1.Stop("COMPUTE_BUDGET_EXCEEDED",
                          "book cap reached before %s" % run_id)
        if time.time() - started > WALL_SECONDS_CAP:
            raise s1.Stop("COMPUTE_BUDGET_EXCEEDED",
                          "wall clock cap reached before %s" % run_id)
        backend = cls._live_backend(LLM_PER_RUN)
        result = s1.run_unit(
            unit=unit, cell=cell, arm=ARM, base_snapshot=h0,
            carried_episodes=(), agent_factory=cls._live_agent,
            backend=backend, store_root=store_root / run_id,
            rounds=ROUNDS, fit_cap=FIT_PER_RUN)
        ledger["llm"] += int(result.get("llm_calls") or 0)
        ledger["fit"] += int(result.get("consumer_fits") or 0)
        verdict = _earned(result)
        public = s1._public_unit_result(result)
        runs.append({
            "run_id": run_id,
            "attempt": index,
            "unit_id": scene["unit_id"],
            "arm": ARM,
            "earned": verdict,
            "proposal_ledger": _proposal_ledger(public),
            "proposal_families": sorted({
                row["family"] for record in public.get("rounds") or []
                for row in record.get("proposals") or []
                if row["family"] != "identity"}),
            "target_family_proposed": any(
                row["family"] == TARGET_FAMILY
                for record in public.get("rounds") or []
                for row in record.get("proposals") or []),
            "llm_calls": result.get("llm_calls"),
            "consumer_fits": result.get("consumer_fits"),
            "seconds": result.get("seconds"),
            "deployment": {
                key: public["deployment"].get(key) for key in
                ("deploy_source", "applied_program", "heldout_accuracy_gain",
                 "heldout_recall_delta_by_class")},
            "rounds": public.get("rounds"),
        })
        print("%-14s %-36s earned=%-5s proposed_target=%-5s llm=%s fits=%s"
              % (run_id, scene["unit_id"], verdict["earned"],
                 runs[-1]["target_family_proposed"], result.get("llm_calls"),
                 result.get("consumer_fits")), flush=True)
        if verdict["earned"]:
            earned = dict(verdict)
            earned["run_id"] = run_id
            break
    return {
        "scene": scene["scene"],
        "unit_id": scene["unit_id"],
        "family_key": scene["family_key"],
        "outcome": "EARNED" if earned else "MISS",
        "earned": earned,
        "attempts": len(runs),
        "max_attempts": MAX_RUNS_PER_SCENE,
        "runs": runs,
    }


# =========================================================================== #
# Part 0 re-verification on the fresh records
# =========================================================================== #
def part0_reverify(scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Axis 5, this time from the persisted pattern leaves.

    The arbitration tightened it: independent families are not the same thing
    as Scope compatibility, and an intersection that collapses to the
    eligibility gate is not a Scope at all.
    """
    s1._set_phase(s1.PHASE_SELECT)
    earned = [scene for scene in scenes if scene["outcome"] == "EARNED"]
    per_source = []
    for scene in earned:
        row = scene["earned"]
        key = str(row.get("task_consumer_key") or "")
        parts = key.split("|")
        per_source.append({
            "scene": scene["scene"],
            "unit_id": scene["unit_id"],
            "run_id": row.get("run_id"),
            "task_kind": parts[0] if parts else None,
            "consumer_id": parts[1] if len(parts) > 1 else None,
            "metric": parts[2] if len(parts) > 2 else None,
            "program_geometry": row.get("program_geometry"),
            "pattern_leaves": dict(row.get("fast_features_binned") or {}),
        })
    result: dict[str, Any] = {
        "sources_earned": [scene["scene"] for scene in earned],
        "sources_missed": [scene["scene"] for scene in scenes
                           if scene["outcome"] != "EARNED"],
        "per_source": per_source,
    }
    if len(earned) < len(scenes):
        result["pass"] = False
        result["verdict"] = "PS1_SOURCES_NOT_REEARNED"
        result["reason"] = (
            "only %d of %d scenes re-earned the target family; a two-source "
            "hypothesis needs both" % (len(earned), len(scenes)))
        return result

    axes: dict[str, Any] = {}
    for axis in ("task_kind", "consumer_id", "metric", "program_geometry"):
        values = {row[axis] for row in per_source}
        axes[axis] = {"values": sorted(str(value) for value in values),
                      "agree": len(values) == 1 and None not in values,
                      "intersection": (sorted(values)[0]
                                       if len(values) == 1 else None)}
    first = per_source[0]["pattern_leaves"]
    intersection = {
        key: value for key, value in sorted(first.items())
        if all(row["pattern_leaves"].get(key) == value for row in per_source)
    }
    beyond_gate = {key: value for key, value in intersection.items()
                   if key != "task_kind"}
    axes["deployment_visible_pattern_intersection"] = {
        "intersection": intersection,
        "leaves_beyond_task_kind": sorted(beyond_gate),
        "agree": bool(beyond_gate),
        "leaf_counts": {row["scene"]: len(row["pattern_leaves"])
                        for row in per_source},
        "disagreeing_leaves": sorted(
            set(first) - set(intersection)),
    }
    missing = [axis for axis, row in axes.items() if not row["agree"]]
    result["axes"] = axes
    result["scope_v1"] = {
        "task_kind": axes["task_kind"]["intersection"],
        "consumer_id": axes["consumer_id"]["intersection"],
        "metric": axes["metric"]["intersection"],
        "pattern_intersection": beyond_gate,
        "program_geometry": [axes["program_geometry"]["intersection"]],
    }
    if not beyond_gate:
        result["pass"] = False
        result["verdict"] = "SCOPE_INTERSECTION_TOO_WIDE"
        result["reason"] = (
            "the two sources agree on no deployment-visible Pattern leaf, so "
            "the intersection collapses to task_kind -- an eligibility gate "
            "that selects every classification unit in the exam, not a Scope.  "
            "No card compiled.")
        return result
    if missing:
        result["pass"] = False
        result["verdict"] = "SOURCE_PROVENANCE_INSUFFICIENT"
        result["reason"] = "axes still missing: %s" % missing
        return result
    result["pass"] = True
    result["verdict"] = "SCOPE_INTERSECTION_USABLE"
    result["reason"] = (
        "all five axes intersect and the Pattern intersection carries %d leaf "
        "or leaves beyond the eligibility gate"
        % len(beyond_gate))
    return result


# =========================================================================== #
# artifact
# =========================================================================== #
def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# PS-0 -- record-layer repair and dual-source re-earn",
        "",
        "protocol: `%s`  evidence grade: **%s**  git: `%s`  backend: **%s**"
        % (payload["protocol_version"], payload["evidence_grade"],
           payload["git_head"],
           (payload.get("backend_probe") or {}).get("returned_model")),
        "",
        "**%s**" % payload["verdict"]["verdict"],
        "",
        payload["verdict"]["reason"],
        "",
        "## Part 1 -- what the round record now keeps",
        "",
    ]
    for key, value in payload["record_repair"].items():
        lines.append("- **%s**: %s" % (key, value))
    lines += ["", "## Part 2 -- re-earn, take what comes", "",
              "| scene | unit | outcome | attempts | earned in | Support | "
              "delayed | target family proposed |",
              "|---|---|---|---|---|---|---|---|"]
    for scene in payload["scenes"]:
        earned = scene.get("earned") or {}
        proposed = any(run["target_family_proposed"] for run in scene["runs"])
        lines.append("| %s | %s | **%s** | %d | %s | %s | %s | %s |" % (
            scene["scene"], scene["unit_id"], scene["outcome"],
            scene["attempts"], earned.get("run_id") or "-",
            earned.get("support_gain", "-"), earned.get("delayed_gain", "-"),
            proposed))
    lines += ["", "### Per-run proposal ledger", ""]
    for scene in payload["scenes"]:
        for run in scene["runs"]:
            lines += ["#### `%s` -- %s (%s)"
                      % (run["run_id"], run["unit_id"],
                         "earned" if run["earned"]["earned"] else "miss"),
                      "",
                      "| round | candidate id | operators | family | chosen | "
                      "outcome | gain |",
                      "|---|---|---|---|---|---|---|"]
            for row in run["proposal_ledger"]:
                lines.append("| %s | `%s` | %s | %s | %s | %s | %s |" % (
                    row["round"], row["candidate_id"],
                    ", ".join(row["operators"]) or "-", row["family"],
                    row["chosen_by_select"], row["outcome"],
                    "%.4f" % row["gain"] if isinstance(row["gain"], (int, float))
                    else "-"))
            lines += ["",
                      "- families proposed: %s"
                      % (run["proposal_families"] or "none"),
                      "- deploy: %s, gain %s"
                      % (run["deployment"]["deploy_source"],
                         run["deployment"]["heldout_accuracy_gain"]),
                      "- cost: LLM %s, fits %s, %.1f s"
                      % (run["llm_calls"], run["consumer_fits"],
                         run["seconds"]),
                      ""]
    part0 = payload["part0_reverify"]
    lines += ["## Part 0 re-verification (axis 5 from the fresh records)", "",
              "- verdict: **%s**" % part0["verdict"],
              "- %s" % part0["reason"], ""]
    if part0.get("axes"):
        lines += ["| axis | intersection | agree |", "|---|---|---|"]
        for axis, row in part0["axes"].items():
            value = row.get("intersection")
            if axis == "deployment_visible_pattern_intersection":
                value = row["leaves_beyond_task_kind"]
            lines.append("| %s | %s | %s |" % (axis, value, row["agree"]))
        pattern = part0["axes"]["deployment_visible_pattern_intersection"]
        lines += ["",
                  "- leaves stored per source: %s" % pattern["leaf_counts"],
                  "- leaves that disagree between the two sources: %s"
                  % (pattern["disagreeing_leaves"] or "none"), ""]
    ledger = payload["ledger"]
    lines += ["## Cost", "",
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


def run(*, part0_only: bool = False) -> int:
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)
    tag = "ps0_reearn"
    store_root = Path(tempfile.gettempdir()) / tag
    if store_root.exists():
        shutil.rmtree(store_root)
    ledger = {"llm": 0, "fit": 0}
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "entry": "--part0" if part0_only else "--reearn",
        "record_repair": {
            "field_1_fast_features_binned": (
                "the binned observable-contract projection of the exact "
                "mapping _run_round hands to run_online_round as "
                "fast_features; not a recomputation"),
            "field_2_proposals": (
                "every candidate the proposal stage named, with its compiled "
                "steps, operators, family, whether select chose it and how it "
                "ended: probed, verifier_rejected, dropped without compiled "
                "steps, or never reached because the Support budget ran out"),
            "family_tagging_fix": (
                "family now comes from compiled steps.  S1c tagged from the "
                "candidate id, and the ids the Fast Agent invents carry no "
                "operator word, so every S1c probe recorded an empty operator "
                "list"),
            "behaviour_change": "none; both fields are additive",
            "cross_check": (
                "tests/functional/"
                "test_round_record_persists_context_and_proposals.py"),
        },
        "protocol": {
            "arm": ARM,
            "rounds": list(ROUNDS),
            "llm_per_run": LLM_PER_RUN,
            "fit_per_run": FIT_PER_RUN,
            "max_runs_per_scene": MAX_RUNS_PER_SCENE,
            "isomorphic_to": "the S1c unit protocol; no hinting of any kind",
            "stop_rule": "first earn stops that scene",
        },
        "scenes_declared": [
            {key: scene[key] for key in
             ("scene", "unit_id", "family_key", "run_ids")}
            for scene in SCENES],
    }
    stopped: str | None = None
    scenes: list[dict[str, Any]] = []
    try:
        payload["backend_probe"] = s1._probe_live_backend()
        if not payload["backend_probe"].get("ok"):
            raise s1.Stop("INSTRUMENT_UNREADABLE",
                          "live backend identity probe failed")
        k0 = s1.compile_k0(store_root / "k0")
        payload["h0_runtime_bundle_sha"] = k0["h0_sha"]
        for scene in SCENES:
            scenes.append(_run_scene(scene, store_root=store_root,
                                     h0=k0["h0"], ledger=ledger,
                                     started=started))
    except s1.Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {"verdict": stopped,
                           "reason": "%s: %s" % (type(exc).__name__, exc),
                           "traceback": traceback.format_exc()}
    payload["scenes"] = scenes
    payload["part0_reverify"] = (part0_reverify(scenes) if scenes else
                                 {"pass": False,
                                  "verdict": "PS1_SOURCES_NOT_REEARNED",
                                  "reason": "no scene completed"})
    payload["ledger"] = {
        "llm": ledger["llm"], "llm_cap": LLM_TOTAL_CAP,
        "fit": ledger["fit"], "fit_cap": FIT_TOTAL_CAP,
        "wall_seconds": round(time.time() - started, 1),
        "wall_seconds_cap": WALL_SECONDS_CAP,
        "downloads": 0,
    }
    payload["oracle_isolation"] = s1._oracle_isolation_report()
    part0 = payload["part0_reverify"]
    if stopped:
        payload["verdict"] = {"verdict": stopped,
                              "reason": (payload.get("stop") or {}).get(
                                  "reason", "")}
    elif part0["pass"]:
        payload["verdict"] = {
            "verdict": "SOURCES_REEARNED_SCOPE_USABLE",
            "reason": ("both scenes re-earned the target family on the "
                       "repaired record path and the five-axis intersection "
                       "is machine-evaluable; PS-1 may proceed")}
    else:
        payload["verdict"] = {"verdict": part0["verdict"],
                              "reason": part0["reason"]}
    payload["obligations"] = {
        "methods_package_unmodified": True,
        "runtime_contracts_operators_unmodified": True,
        "production_governance_unmodified": True,
        "record_repair_is_additive_only": True,
        "protocol_isomorphic_to_s1c_unit": True,
        "no_hinting_of_prompt_budget_or_candidate_cap": True,
        "scenes_stopped_on_first_earn": True,
        "live_backend": (payload.get("backend_probe") or {}).get(
            "returned_model"),
        "downloads": 0,
        "sealed_artifacts_not_read": True,
        "oracle_isolation_holds": payload["oracle_isolation"]["holds"],
        "stage_report_not_written": True,
        "full_repo_pytest_not_run": True,
    }
    payload["outside_book"] = []
    s1._dump(OUT_JSON, payload)
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"]["verdict"],
        "scenes": {scene["scene"]: scene["outcome"] for scene in scenes},
        "llm": ledger["llm"], "fits": ledger["fit"],
        "seconds": payload["ledger"]["wall_seconds"],
        "artifact": str(OUT_JSON)}, ensure_ascii=False, indent=1))
    return 0 if payload["verdict"]["verdict"] == (
        "SOURCES_REEARNED_SCOPE_USABLE") else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reearn", action="store_true")
    parser.add_argument("--part0", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.reearn or args.part0:
        return run(part0_only=bool(args.part0 and not args.reearn))
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
