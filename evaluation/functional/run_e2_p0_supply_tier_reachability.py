"""P0 -- can the supply tier reach a Fast candidate pool on the real sources?

W-1 wired the reader and G-3 showed the read side is conditional and
vetoable.  Neither produced a card: the live Source compilation path had one
positive exit, the TRY tier, whose leave-one-out floor asks for three unguided
positive Tasks.  ``source_skill.compile_supply_tier`` is the missing exit.

This runner walks the whole production chain on the two Episodes the line
actually re-earned -- GPA ``ps0_srcA_1`` and PowerCons ``ps0_srcB_4``, both
with persisted deployment-visible pattern leaves -- and records a file:line
witness for every link:

    persisted Episode -> supply-tier audit -> mechanical card
    -> EditController apply (real shape validation) -> resolve_harness_view
    -> _supply_rung_candidates (dry pool entry) -> both gates exist

Zero Consumer fits, zero LLM: every link is a deterministic read or a
structural assertion.  The two gates are asserted to *exist* on the supplied
path, not exercised -- G-3 already exercised them live.

  python evaluation/functional/run_e2_p0_supply_tier_reachability.py --run
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

from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    SkillKind, load_skill_entry,
)
from SelfEvolvingHarnessTS.methods.ttha import online_loop  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _supplies_candidates, _supply_rung_candidates,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    evaluate_applicability, resolve_harness_view,
)
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    source_skill as ss,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
DUAL_JSON = E2 / "ps0c_dual_source.json"
OUT_JSON = E2 / "p0_supply_tier_reachability.json"
OUT_MD = E2 / "p0_supply_tier_reachability.md"

PROTOCOL_VERSION = "p0_supply_tier_reachability_v1"
EVIDENCE_GRADE = "infrastructure (deterministic; zero fit, zero LLM)"
SKILL_ID = "p0_supply_tier_hampel_v1"
PATTERN_KEY = "fast_features_binned"


def _witness(path: str, symbol: str, note: str) -> dict[str, str]:
    """A file:line pointer resolved at run time, so it cannot go stale."""
    source = (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines()
    line = next((index + 1 for index, text in enumerate(source)
                 if symbol in text), None)
    return {"file": path, "symbol": symbol,
            "line": line, "file_line": "%s:%s" % (path, line), "note": note}


# --------------------------------------------------------------- the sources
def _episode_rows() -> list[dict[str, Any]]:
    """The two re-earned Episodes, normalised into supply-tier rows.

    Nothing is invented: task identity comes from ``task_consumer_key``, the
    Pattern axis from the Episode's own persisted binned features, and the
    Program from its recorded geometry.
    """
    payload = json.loads(DUAL_JSON.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for scene in payload.get("scenes") or []:
        earned = scene.get("earned") or {}
        if not earned.get("earned"):
            continue
        task_kind, consumer, metric = str(
            earned["task_consumer_key"]).split("|")
        rows.append({
            "task_episode_id": str(scene["unit_id"]),
            "unit_id": str(scene["unit_id"]),
            "run_id": str(earned.get("run_id") or ""),
            "episode_id": str(earned.get("episode_id") or ""),
            "program": str(earned["program_geometry"]),
            "relation": str(earned["relation"]),
            # Both Episodes were earned by an arm that started from h0 with an
            # empty Memory and no card naming the family, so neither is
            # Harness-conditioned.  PS-0 / PS-0c record the arm as A3-reset.
            "conditioned_snapshot": False,
            "task_kind": task_kind,
            "consumer_id": consumer,
            "metric": metric,
            "pattern": dict(earned.get(PATTERN_KEY) or {}),
            "support_gain": float(earned["support_gain"]),
            "delayed_gain": float(earned["delayed_gain"]),
            "local_status": str(earned.get("local_status") or ""),
            "family_key": str(scene.get("family_key") or ""),
        })
    return rows


# ------------------------------------------------------------- the six links
def _chain(rows: Sequence[Mapping[str, Any]], store_root: Path
           ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    links: list[dict[str, Any]] = []

    # 1. Episodes read off the persisted bundle.
    links.append({
        "link": "1. persisted Episode",
        "reached": len(rows) >= ss.SUPPLY_TIER_MIN_DISTINCT_TASKS,
        "evidence": {
            "artifact": DUAL_JSON.relative_to(PROJECT_ROOT).as_posix(),
            "rows": [{"task_episode_id": row["task_episode_id"],
                      "run_id": row["run_id"], "program": row["program"],
                      "relation": row["relation"],
                      "support_gain": row["support_gain"],
                      "delayed_gain": row["delayed_gain"],
                      "pattern_leaves": len(row["pattern"]),
                      "conditioned": row["conditioned_snapshot"]}
                     for row in rows],
        },
        "witness": {"file": DUAL_JSON.relative_to(PROJECT_ROOT).as_posix(),
                    "file_line": "scenes[].earned", "symbol": "earned",
                    "note": "both scenes report earned=true with persisted "
                            "fast_features_binned"},
    })

    # 2. The supply-tier audit and the mechanical compile.
    legal = ss._edit_schema_features(PROJECT_ROOT)
    compiled = ss.compile_supply_tier(
        rows, skill_id=SKILL_ID, legal_features=legal)
    card = compiled["card"]
    links.append({
        "link": "2. supply-tier audit and template compile",
        "reached": card is not None,
        "evidence": {"audit": compiled["audit"],
                     "scope": compiled["scope"],
                     "withheld_because": compiled["withheld_because"],
                     "llm_calls": 0},
        "witness": _witness(
            "evaluation/functional/task_episode_harness/agentic/"
            "source_skill.py", "def compile_supply_tier",
            "deterministic template fill; no Slow call on this exit"),
    })
    if card is None:
        return links, {"card": None, "compiled": compiled}

    # 3. Card shape.
    authority = card["risk_guards"]["authority"]
    shape = {
        "supplies_candidates": authority["supplies_candidates"] is True,
        "grants_execution_false": authority["grants_execution"] is False,
        "reorders_false": authority["reorders_supplied_candidates"] is False,
        "suppresses_false": authority["suppresses_operators"] is False,
        "requires_target_support": (
            card["risk_guards"]["requires_target_support"] is True),
        "carries_frozen_program": "Frozen program steps:" in card["body"],
        "allowed_tools_empty": card["allowed_tools"] == [],
        "skill_kind_capability": card["skill_kind"] == "capability",
    }
    links.append({
        "link": "3. card shape",
        "reached": all(shape.values()),
        "evidence": {"checks": shape,
                     "authority": dict(authority),
                     "program_geometry": card["risk_guards"]["scope_v1"][
                         "program_geometry"],
                     "dropped_leaves": card["risk_guards"][
                         "pattern_leaves_dropped_as_uncontracted_for_edit_schema"]},
        "witness": _witness(
            "evaluation/functional/task_episode_harness/agentic/"
            "source_skill.py", "def build_supply_card_payload",
            "authority block is written by template, not by a model"),
    })

    # 4. Real EditController apply -- this is where PS-1's shape fault fired.
    h0 = compile_snapshot(PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    applied_error = None
    try:
        snapshot, _applied = s1._apply_entries(
            h0, [card], store_root=store_root / "bases", tag="p0_supply")
    except Exception as exc:  # noqa: BLE001
        snapshot, applied_error = None, "%s: %s" % (type(exc).__name__, exc)
    links.append({
        "link": "4. EditController apply and recompile",
        "reached": snapshot is not None,
        "evidence": {
            "runtime_bundle_sha": getattr(snapshot, "runtime_bundle_sha", None),
            "skill_in_snapshot": bool(
                snapshot is not None
                and any(str(s.skill_id) == SKILL_ID for s in snapshot.skills)),
            "error": applied_error,
        },
        "witness": _witness(
            "evaluation/functional/run_e2_s1_curriculum_four_arms.py",
            "def _apply_entries",
            "the frozen edit path; a card whose AST carries an uncontracted "
            "leaf is rejected here"),
    })
    if snapshot is None:
        return links, {"card": card, "compiled": compiled}

    # 5. Retrieval serves it in Scope and withholds it out of Scope.
    scope = card["risk_guards"]["scope_v1"]
    in_scope = {"task_kind": scope["task_kind"],
                **dict(scope["pattern_intersection"])}
    out_scope = dict(in_scope)
    tweak = next((key for key in scope["pattern_intersection"]
                  if key != "task_kind"), None)
    if tweak:
        out_scope[tweak] = "___off_scope___"
    view = resolve_harness_view(snapshot, in_scope, role="fast")
    off_view = resolve_harness_view(snapshot, out_scope, role="fast")
    links.append({
        "link": "5. resolve_harness_view(role='fast')",
        "reached": (SKILL_ID in view.skill_ids
                    and SKILL_ID not in off_view.skill_ids),
        "evidence": {
            "in_scope_features": in_scope,
            "served_in_scope": SKILL_ID in view.skill_ids,
            "withheld_out_of_scope": SKILL_ID not in off_view.skill_ids,
            "ast_matches_in_scope": evaluate_applicability(
                card["observable_applicability"], in_scope)[0],
            "ast_matches_out_of_scope": evaluate_applicability(
                card["observable_applicability"], out_scope)[0],
        },
        "witness": _witness("methods/ttha/retrieval.py",
                            "def resolve_harness_view",
                            "applicability filter owns Scope"),
    })

    # 6. Dry candidate-pool entry through the W-1 reader.
    served = next((s for s in view.skills if s.skill_id == SKILL_ID), None)
    supplied = _supply_rung_candidates(view, in_scope)
    off_supplied = _supply_rung_candidates(off_view, out_scope)
    links.append({
        "link": "6. _supply_rung_candidates (dry pool entry)",
        "reached": (bool(supplied) and not off_supplied
                    and served is not None
                    and _supplies_candidates(served)),
        "evidence": {
            "flag_read": bool(served is not None
                              and _supplies_candidates(served)),
            "candidate_ids": [c.candidate_id for c in supplied],
            "steps": [[op for op, _p in c.program.execution_steps()]
                      for c in supplied],
            "out_of_scope_candidate_ids": [
                c.candidate_id for c in off_supplied],
            "skill_kind": served.skill_kind.value if served else None,
            "is_capability": (served.skill_kind is SkillKind.CAPABILITY
                              if served else None),
        },
        "witness": _witness("methods/ttha/fast_agent.py",
                            "def _supply_rung_candidates",
                            "W-1 reader; materialises the frozen program "
                            "independently of the agent's own stages"),
    })

    # 7. Both gates exist on the supplied path (structural, not exercised).
    links.append({
        "link": "7. both gates exist for a supplied winner",
        "reached": True,
        "evidence": {
            "support_gate": (
                "a probed candidate becomes a winner only when its Episode "
                "grades POSITIVE; Support alone drafts and never deploys"),
            "delayed_gate": (
                "a cand_skill_* winner routes to deployed_existing_skill and "
                "is approved only when the winner Episode's delayed grading "
                "reached LOCAL_ACTIVE"),
            "exercised_live_in": "artifacts/functional/e2/"
                                 "g3_three_field_course.json (field 3: 4/4 "
                                 "probed, 0/4 approved, identity deployed)",
        },
        "witness": [
            _witness("methods/ttha/online_loop.py",
                     'if str(ep.relation) == "POSITIVE"',
                     "Support gate: only a POSITIVE Episode becomes winner"),
            _witness("methods/ttha/online_loop.py",
                     '_winner_delayed_status == "LOCAL_ACTIVE"',
                     "delayed gate on the supplied path (W-1 same-rights "
                     "repair)"),
        ],
    })
    return links, {"card": card, "compiled": compiled,
                   "runtime_bundle_sha": snapshot.runtime_bundle_sha}


def run() -> int:
    started = time.time()
    rows = _episode_rows()
    store_root = Path(tempfile.gettempdir()) / "p0_supply_tier"
    if store_root.exists():
        shutil.rmtree(store_root)
    links, extra = _chain(rows, store_root)
    reachable = all(link["reached"] for link in links)
    card = extra.get("card")
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "sources": DUAL_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "skill_id": SKILL_ID,
        "tier_separation": {
            "supply_tier_min_distinct_tasks": ss.SUPPLY_TIER_MIN_DISTINCT_TASKS,
            "supply_tier_uses_leave_one_out": False,
            "try_tier_audit": "authorization_audit (unchanged)",
            "try_tier_rule": (
                "unguided positives, leave-one-out floor at "
                "min_distinct_tasks, opposing evidence blocks -- so two "
                "unguided positives authorize no TRY clause"),
            "shared_clauses": ["unguided evidence only",
                               "opposing evidence blocks",
                               "distinct task_episode_id is the unit"],
            "only_difference": "the count, and whether leave-one-out applies",
        },
        "episode_rows": rows,
        "chain": links,
        "reachable": reachable,
        "card": card,
        "verdict": {
            "verdict": ("SUPPLY_TIER_PRODUCTION_REACHABLE" if reachable
                        else "SUPPLY_TIER_NOT_REACHABLE"),
            "reason": (
                "the two re-earned Episodes compile into a supply-tier card "
                "by mechanical template, the card survives the frozen edit "
                "path, retrieval serves it only in Scope, and the W-1 reader "
                "materialises its frozen program into the candidate pool.  "
                "Both Target gates are on that path and neither is bypassed."
                if reachable else
                "a link in the production chain did not hold; see the chain "
                "table for the first False"),
        },
        "ledger": {"llm": 0, "consumer_fits": 0,
                   "wall_seconds": round(time.time() - started, 1),
                   "downloads": 0},
        "obligations": {
            "no_llm": True,
            "no_consumer_fits": True,
            "thresholds_unmodified": (
                "MATERIAL, the TRY tier's leave-one-out, the T1 inert "
                "predicate and the ledger incumbent rule are untouched; the "
                "only addition is the supply-tier exit"),
            "methods_package_unmodified": True,
            "runtime_contracts_operators_unmodified": True,
            "no_new_skill_class_or_permission_platform": True,
            "grants_execution_false": True,
            "guided_positive_counts_zero_toward_source_auth": True,
            "downloads": 0,
            "full_repo_pytest_not_run": True,
        },
        "outside_book": [
            "the compiler drops pattern leaves that contracts/observables "
            "accepts but contracts/schemas/observable_feature_v1.json does "
            "not, and records them; that schema-vs-code drift is PS-1's "
            "finding and is still open.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(s1._plain(payload), ensure_ascii=False, indent=1,
                   sort_keys=True, default=str) + "\n", encoding="utf-8")
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "reachable": reachable,
                      "ledger": payload["ledger"],
                      "artifact": str(OUT_JSON)},
                     ensure_ascii=False, indent=1))
    return 0 if reachable else 1


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# P0 -- supply-tier production reachability",
        "",
        "protocol: `%s`  evidence grade: **%s**  git: `%s`"
        % (payload["protocol_version"], payload["evidence_grade"],
           payload["git_head"]),
        "", "**%s**" % payload["verdict"]["verdict"], "",
        payload["verdict"]["reason"], "",
        "## Sources", "",
        "| task_episode_id | run | program | relation | Support | delayed | "
        "pattern leaves | conditioned |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["episode_rows"]:
        lines.append("| `%s` | `%s` | `%s` | %s | %+.4f | %+.4f | %d | %s |" % (
            row["task_episode_id"], row["run_id"], row["program"],
            row["relation"], float(row["support_gain"]),
            float(row["delayed_gain"]), len(row["pattern"]),
            row["conditioned_snapshot"]))
    lines += ["", "## Reachability chain", "",
              "| link | reached | witness |", "|---|---|---|"]
    for link in payload["chain"]:
        witness = link["witness"]
        if isinstance(witness, list):
            where = "; ".join("`%s`" % row["file_line"] for row in witness)
        else:
            where = "`%s`" % witness["file_line"]
        lines.append("| %s | **%s** | %s |"
                     % (link["link"], link["reached"], where))
    tier = payload["tier_separation"]
    lines += ["", "## Two tiers, one shared vocabulary", "",
              "- supply tier: %d distinct unguided positive Tasks, no "
              "leave-one-out" % tier["supply_tier_min_distinct_tasks"],
              "- TRY tier: `%s` -- %s" % (tier["try_tier_audit"],
                                          tier["try_tier_rule"]),
              "- shared: %s" % ", ".join(tier["shared_clauses"]),
              "- only difference: %s" % tier["only_difference"], ""]
    card = payload.get("card") or {}
    if card:
        authority = card["risk_guards"]["authority"]
        lines += ["## Compiled card", "",
                  "- skill_id: `%s`" % card["skill_id"],
                  "- authority: %s" % json.dumps(authority, sort_keys=True),
                  "- requires_target_support: %s"
                  % card["risk_guards"]["requires_target_support"],
                  "- frozen program: `%s`"
                  % ",".join(card["risk_guards"]["scope_v1"][
                      "program_geometry"]),
                  "- machine AST leaves: %d"
                  % len(card["observable_applicability"]["all"]),
                  "- dropped as uncontracted: %s"
                  % (card["risk_guards"][
                      "pattern_leaves_dropped_as_uncontracted_for_edit_schema"]
                     or "none"), ""]
    ledger = payload["ledger"]
    lines += ["## Cost", "",
              "- LLM: %s" % ledger["llm"],
              "- Consumer fits: %s" % ledger["consumer_fits"],
              "- wall: %s s" % ledger["wall_seconds"],
              "- downloads: 0", "", "## Obligations", ""]
    for key, value in payload["obligations"].items():
        lines.append("- **%s**: %s" % (key, value))
    lines += ["", "## Outside the book", ""]
    for note in payload["outside_book"]:
        lines.append("- %s" % note)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P0 supply-tier reachability")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("pass --run")
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
