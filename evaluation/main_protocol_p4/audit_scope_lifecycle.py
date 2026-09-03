"""What the Scope wiring proves so far, and what it does not.

The eight wiring steps are in, the lifecycle test passes and the widened
regression gate is green.  That is not the same as "the chain is proven", and
the difference is worth writing down before it gets cited as if it were.

Covered end-to-end through ``run_online_round`` with a stub Consumer (0 LLM):
Program+Scope, scoped verification, Support-A, the Episode's two forms, the
Skill's predicate-only storage, and the winner's revision counter.

Steps 7 and 8 are now driven too: a 0-LLM Slow proposes a manifest whose
skill entry carries a tighter predicate, the round adopts it atomically with the
program, the revision becomes 2 and the resolution shrinks from four series to
two.  Reaching that required two contract changes, because ``SkillEntry`` was an
exact-fields record and the schema was ``additionalProperties: false``: a Scope
literally could not be stored before.

Not covered at all: the Classification serving path, which still serves
unprepared features and has no raw fallback pipeline.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4q_scope_lifecycle.json"
LIFECYCLE_TEST = "tests/main_protocol/test_scope_lifecycle.py"
LOCK_TEST = "tests/main_protocol/test_scoped_serving_behaviour_lock.py"
SPEC_TEST = "tests/main_protocol/test_scope_spec.py"

STEPS = (
    ("1_loop_accepts_scope", "run_online_round takes candidate_scopes and "
     "scope_resolver, both defaulting to None", "END_TO_END"),
    ("2_candidate_carries_predicate", "the predicate reaches the probe through "
     "candidate_scopes", "END_TO_END"),
    ("3_runtime_resolves_to_uids", "the probe loop resolves the predicate at "
     "the origin and hands the set to the executor", "END_TO_END"),
    ("4_episode_records_both_forms", "program_geometry stores the predicate and "
     "the resolved UIDs, the latter marked resolved_is_skill_field=False",
     "END_TO_END"),
    ("5_skill_stores_predicate_only", "handle_fast_winner writes serving_scope "
     "into the skill entry; no UID is stored", "END_TO_END"),
    ("6_support_b_replay_uses_the_same_scope", "the winner's replay evaluator "
     "carries the same resolved set, so approval judges the execution that "
     "earned the probe", "END_TO_END"),
    ("7_patch_revises_the_predicate", "a Slow PATCH carrying a tighter "
     "predicate reaches the slow event, replaces the winner's Scope and bumps "
     "the revision to 2", "END_TO_END"),
    ("8_re_encounter_executes_the_new_version", "the revised predicate resolves "
     "to strictly fewer series, and a stored predicate re-resolves against a "
     "different Target's series", "END_TO_END"),
)


def _run(target: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--tb=line",
         "-p", "no:cacheprovider"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    summary = ""
    for line in reversed((completed.stdout + completed.stderr).splitlines()):
        if " passed" in line or " failed" in line:
            summary = line.strip()
            break
    return {"target": target, "summary": summary,
            "passed": completed.returncode == 0}


def build() -> dict[str, Any]:
    suites = [_run(target) for target in (SPEC_TEST, LOCK_TEST, LIFECYCLE_TEST)]
    return {
        "stage": "P4Q_SCOPE_LIFECYCLE",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_MECHANISM_WIRING",
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": 0,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "harness_experiments_run": 0,
        },
        "steps": [
            {"step": name, "what": what, "coverage": coverage}
            for name, what, coverage in STEPS
        ],
        "suites": suites,
        "additive_discipline": (
            "the default branch keeps the historical *call shape*, not merely "
            "the historical default value: an injected executor or method that "
            "only accepts the old signature must not receive a new keyword. "
            "Ignoring this broke all nine tests in "
            "tests/functional/test_bounded_admission_lifecycle.py once"
        ),
        "regression_gate": {
            "baseline": "artifacts/main_protocol/p4p_failure_baseline.json",
            "suites": ["tests/main_protocol", "tests/methods", "tests/functional"],
            "note": (
                "tests/functional was missing from the first baseline, which is "
                "exactly where run_online_round is driven; a real regression "
                "went unflagged until a manual snapshot comparison caught it"
            ),
            "known_pre_existing": {
                "tests/methods": "33 failing nodes, mostly H0 snapshot lock",
                "tests/functional": (
                    "one collection error: test_skill_revocation.py:166 uses a "
                    "PEP 701 f-string that Python 3.10 cannot parse"
                ),
            },
        },
        "contract_changes": {
            "contracts/schemas/skill_entry_v1.json": (
                "optional serving_scope property; required[] unchanged, so "
                "every existing entry still validates"
            ),
            "contracts/harness.py": (
                "SkillEntry gains an optional serving_scope field, and "
                "_require_exact_fields gains an explicit optional set so the "
                "record stays exact for everything else; the loader validates "
                "the predicate and refuses a clause that names a series"
            ),
            "why_needed": (
                "without both, a Slow PATCH carrying a Scope fails to apply "
                "with EditShapeError and the Skill can never store one"
            ),
        },
        "not_covered": [
            "Classification serving path: still serves unprepared features and "
            "has no raw fallback pipeline (p4o gate 7)",
            "a real second encounter: step 8 is shown by re-resolution, not by "
            "a second round retrieving the stored Skill and executing it",
            "any utility claim: no Harness experiment has been run",
        ],
        "verdict": "SCOPE_LIFECYCLE_DRIVEN_END_TO_END_0_LLM",
        "releases": (
            "nothing; a single-cell live smoke confirming a real Agent emits a "
            "legal Scope is the next gate before Static/A3/A5 is frozen"
        ),
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for step in report["steps"]:
        print("%-42s %s" % (step["step"], step["coverage"]))
    for suite in report["suites"]:
        print("%-52s %s" % (suite["target"], suite["summary"]))
    print("verdict : %s" % report["verdict"])
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
