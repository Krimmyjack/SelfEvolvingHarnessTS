"""Did a *scoped* Skill actually survive?  Four checks, read-only.

sol's adjudication on the clean replicate names exactly what has to hold before
the run may be read as a confirmation, and each of the four is a place this
protocol has already been wrong:

1. **the Scope reached the disk** -- ``skill_entry_to_dict`` dropped
   ``serving_scope`` for the whole life of the SCOPE work, so a Skill gated as
   "this program on these series" persisted as "this program".  Checked against
   the snapshot store, not against the in-memory round;
2. **the delayed gate passed** -- on a window the revision was carried to, with
   the predicate re-resolved there rather than replayed as a UID list;
3. **an independent re-encounter passed** -- at a later held-in origin the
   revision did not help produce;
4. **the stored Skill still carries the same predicate** -- the one the two
   gates measured, clause for clause.

Check 4 is the one that would have caught the p4w3 defect on its own: there,
1 was false while 2 and 3 were true, and the verdict string said the Skill had
survived.  A verdict that reads only the round is a verdict about something the
store never held.

Read-only: this opens artifacts and the snapshot store and writes one report.
It takes no reading of its own and touches no held-out data.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "artifacts/main_protocol"


def _out_for(artifact: Path) -> Path:
    """Named after what it audits, so two audits cannot land on one path."""
    return AUDIT_DIR / ("p4w3c_scoped_skill_survival.%s.json" % artifact.stem)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _clause(item: Any) -> dict[str, Any] | None:
    """One clause, from either the current shape or the older broken one.

    Runs before the recording fix wrote frozen predicates through
    ``json default=str``, so their clauses are stored as ``repr`` strings such
    as ``"{'feature': 'missing_fraction', 'op': '<', 'threshold': 0.2}"``.
    Those artifacts are evidence and are not being rewritten, so the audit
    reads them rather than refusing them.
    """
    if isinstance(item, Mapping):
        return _plain(dict(item))
    if isinstance(item, str):
        import ast  # noqa: PLC0415 - only needed for the legacy shape
        try:
            parsed = ast.literal_eval(item)
        except (ValueError, SyntaxError):
            return None
        return _plain(dict(parsed)) if isinstance(parsed, Mapping) else None
    return None


def _clauses(scope: Mapping[str, Any] | None) -> list[dict[str, Any]] | None:
    if not scope:
        return None
    parsed = [c for c in (_clause(item) for item in
                          (scope.get("predicate") or ())) if c is not None]
    return sorted(parsed,
                  key=lambda c: (str(c.get("feature")), str(c.get("op"))))


def persisted_probe(round_row: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The predicate a round actually deployed, in comparable form."""
    return _clauses(round_row.get("winner_serving_scope"))


def _stored_skills(store_root: Path) -> dict[str, dict[str, Any]]:
    """Every Skill the store actually holds, newest snapshot last."""
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(store_root.rglob("skills/learned/*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entry["_path"] = path.relative_to(PROJECT_ROOT).as_posix()
        found[str(entry.get("skill_id"))] = entry
    return found


def audit(artifact: Path, store_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stage": "P4W3C_SCOPED_SKILL_SURVIVAL",
        "written_at": datetime.now().astimezone().isoformat(),
        "reads": {"artifact": artifact.name, "store": store_root.name},
        "is_read_only": True,
    }
    if not artifact.exists():
        report["status"] = "ARTIFACT_MISSING"
        report["verdict"] = "CANNOT_AUDIT"
        return report
    run = json.loads(artifact.read_text(encoding="utf-8"))
    report["run_label"] = run.get("run_label")
    report["run_verdict"] = run.get("verdict")
    report["llm_calls"] = run.get("llm_calls")
    report["run_ledger"] = run.get("run_ledger")

    stored = _stored_skills(store_root)
    activated = list(run.get("activated_skill_ids") or ())
    report["activated_skill_ids"] = activated
    report["skills_in_store"] = sorted(stored)

    findings = []
    for skill_id in activated:
        entry = stored.get(skill_id)
        # Which round activated it, and what the gates there measured.  Matched
        # on the predicate the round deployed when the store has one, so a run
        # that activated more than once cannot silently be audited against the
        # wrong round's readings.
        activating = [r for r in (run.get("rounds") or ()) if r.get("activated")]
        round_row = next(
            (r for r in activating
             if persisted_probe(r) == _clauses((entry or {}).get("serving_scope"))),
            activating[0] if activating else None)
        delayed = (round_row or {}).get("delayed_gate") or {}
        reenc = (round_row or {}).get("re_encounter_gate") or {}
        gated = _clauses((round_row or {}).get("winner_serving_scope"))
        persisted = _clauses((entry or {}).get("serving_scope"))

        checks = {
            # Two different failures, and conflating them would overstate the
            # evidence: a Skill absent from the store audited says nothing on
            # its own about whether the writer kept the Scope.
            "found_in_the_store_audited": entry is not None,
            "scope_persisted_to_disk": bool(entry and entry.get("serving_scope")),
            "delayed_gate_passed": bool(delayed.get("passes")),
            "independent_re_encounter_passed": bool(reenc.get("passes")),
            "stored_predicate_matches_the_gated_one": (
                gated is not None and persisted is not None
                and gated == persisted),
        }
        findings.append({
            "skill_id": skill_id,
            "found_in_store": entry is not None,
            "store_path": (entry or {}).get("_path"),
            "activated_at_origin": (round_row or {}).get("origin"),
            "gated_predicate": gated,
            "persisted_predicate": persisted,
            "delayed_reading": {k: delayed.get(k) for k in
                                ("read_origin", "treated", "served",
                                 "aggregate_gain", "harmed_fraction",
                                 "max_single_series_harm", "passes",
                                 "failed_lines")},
            "re_encounter_reading": {k: reenc.get(k) for k in
                                     ("read_origin", "treated", "served",
                                      "aggregate_gain", "harmed_fraction",
                                      "max_single_series_harm", "passes",
                                      "failed_lines")},
            "checks": checks,
            "all_four_hold": all(checks.values()),
        })

    report["skills"] = findings
    survived = [f for f in findings if f["all_four_hold"]]
    report["scoped_survivors"] = [f["skill_id"] for f in survived]
    # Deliberately narrow wording.  One trajectory is a confirmation that the
    # lifecycle can complete, and it is not a success rate: the two v3 runs
    # already diverged on the same origins because the Fast proposal is not
    # deterministic.  Repetition belongs to the pre-registered Phase S course.
    report["verdict"] = (
        "SCOPED_SKILL_SURVIVED_ONCE" if survived
        else "NO_SCOPED_SKILL_SURVIVED" if findings
        else "NO_SKILL_ACTIVATED")
    report["what_this_does_not_claim"] = (
        "a success rate.  This is one trajectory on one cohort and one origin "
        "order; the live Fast proposal is stochastic and two v3 runs already "
        "diverged on these same five origins.  Replication is Phase S's, with "
        "different cohorts and a pre-registered number of trajectories"
    )
    report["boundary"] = {"held_out_reads": 0, "artifacts_overwritten": 0,
                          "arms_touched": 0}
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        default="artifacts/main_protocol/"
                "p4w3b_source_line_v3_clean_post_fix_replicate_1.json")
    parser.add_argument("--store", default=".p4w3_source_store")
    args = parser.parse_args(argv)
    artifact = PROJECT_ROOT / args.artifact
    report = audit(artifact, PROJECT_ROOT / args.store)
    OUT = _out_for(artifact)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print("verdict        : %s" % report.get("verdict"))
    for finding in report.get("skills", ()):
        print("  %-52s %s" % (
            finding["skill_id"],
            " ".join("%s=%s" % (name.split("_")[0], "OK" if ok else "NO")
                     for name, ok in finding["checks"].items())))
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
