"""Two safety measures, taken before the Scope wiring touches a core module.

The working tree is not a safe place to edit from: ``git status`` shows 833
entries, ``methods/ttha/online_loop.py``, ``fast_agent.py`` and ``method.py``
were already modified before this line of work began, and several core modules
(``admission_policy.py``, ``ordering_card.py``, ``p4_runner.py``) are untracked
altogether.  ``git checkout`` therefore cannot restore them.

**Snapshot.**  Every file the Scope wiring may touch is copied under
``artifacts/main_protocol/_scope_integration_snapshot/`` with a SHA-256 manifest,
so any edit is reversible by file rather than by version control.  ``--restore``
puts them back and reports what changed.

**Failure baseline.**  ``tests/methods`` already fails.  Recording only the
count would let a new failure hide behind an old one, so the baseline stores
each node id with a normalised reason, and ``--compare`` reports three things
separately: nodes that newly fail, nodes that stopped failing, and nodes whose
reason changed.  The gate after any edit is: ``tests/main_protocol`` stays
green, ``tests/methods`` gains no new failing node, and no existing reason
degrades.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = PROJECT_ROOT / "artifacts/main_protocol/_scope_integration_snapshot"
MANIFEST = SNAPSHOT / "manifest.json"
BASELINE = PROJECT_ROOT / "artifacts/main_protocol/p4p_failure_baseline.json"

#: Everything the eight wiring steps could plausibly reach.
TRACKED = (
    "methods/ttha",
    "tests/methods",
    "tests/main_protocol",
    "tests/functional",
    "evaluation/main_protocol_p4/scoped_serving_evaluator.py",
    "evaluation/main_protocol_p4/scope_spec.py",
)
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files() -> list[Path]:
    found: list[Path] = []
    for entry in TRACKED:
        root = PROJECT_ROOT / entry
        if root.is_file():
            found.append(root)
            continue
        for path in sorted(root.rglob("*")):
            if path.is_dir() or any(part in SKIP_DIRS for part in path.parts):
                continue
            found.append(path)
    return found


def take_snapshot() -> dict[str, Any]:
    if SNAPSHOT.exists():
        shutil.rmtree(SNAPSHOT)
    entries = {}
    for path in _files():
        relative = path.relative_to(PROJECT_ROOT)
        target = SNAPSHOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        entries[relative.as_posix()] = {
            "sha256": _digest(path), "bytes": path.stat().st_size,
        }
    payload = {
        "stage": "SCOPE_INTEGRATION_SNAPSHOT",
        "taken_at": datetime.now().astimezone().isoformat(),
        "why": (
            "git cannot restore these: several are untracked and the tree has "
            "833 uncommitted entries predating this work"
        ),
        "roots": list(TRACKED),
        "file_count": len(entries),
        "files": entries,
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def compare_snapshot() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed, missing, added = [], [], []
    for relative, entry in payload["files"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            missing.append(relative)
        elif _digest(path) != entry["sha256"]:
            changed.append(relative)
    known = set(payload["files"])
    for path in _files():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative not in known:
            added.append(relative)
    return {"changed": sorted(changed), "missing": sorted(missing),
            "added": sorted(added)}


def restore_snapshot(only: Sequence[str] | None = None) -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    restored = []
    for relative in payload["files"]:
        if only and relative not in set(only):
            continue
        source = SNAPSHOT / relative
        target = PROJECT_ROOT / relative
        if not source.is_file():
            continue
        if target.is_file() and _digest(target) == _digest(source):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        restored.append(relative)
    return {"restored": sorted(restored)}


_REASON = re.compile(r"^(E\s+)?(?P<body>[A-Za-z_.]*(Error|Exception|Failed)[^\n]*)")


def _normalise(reason: str) -> str:
    """Collapse a traceback tail to a comparable cause."""
    text = " ".join(reason.split())
    for pattern, label in (
        (r"snapshot lock mismatch", "H0_SNAPSHOT_LOCK_MISMATCH"),
        (r"ModuleNotFoundError", "MODULE_NOT_FOUND"),
        (r"FileNotFoundError", "FILE_NOT_FOUND"),
        (r"AssertionError", "ASSERTION"),
        (r"TypeError", "TYPE_ERROR"),
        (r"ValueError", "VALUE_ERROR"),
        (r"KeyError", "KEY_ERROR"),
        (r"AttributeError", "ATTRIBUTE_ERROR"),
    ):
        if re.search(pattern, text):
            return label
    return "OTHER"


def run_suite(target: str) -> dict[str, Any]:
    completed = subprocess.run(
        # One un-importable file must not hide a whole suite: tests/functional
        # contains a pre-existing SyntaxError (PEP 701 f-string on Python 3.10)
        # that otherwise aborts collection and silently reports zero failures.
        [sys.executable, "-m", "pytest", target, "-q", "--tb=line",
         "--continue-on-collection-errors", "-p", "no:cacheprovider"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    output = completed.stdout + completed.stderr
    failures: dict[str, str] = {}
    collection_errors: list[str] = []
    for line in output.splitlines():
        if line.startswith("ERROR "):
            collection_errors.append(line[len("ERROR "):].strip())
        if line.startswith("FAILED "):
            node = line[len("FAILED "):].split(" - ")[0].strip()
            tail = line.split(" - ", 1)[1] if " - " in line else ""
            failures[node] = _normalise(tail)
        elif "Error" in line and "::" in line and line.strip().startswith("/"):
            continue
    summary = ""
    for line in reversed(output.splitlines()):
        if " passed" in line or " failed" in line:
            summary = line.strip()
            break
    return {
        "target": target,
        "failing_nodes": dict(sorted(failures.items())),
        "failing_count": len(failures),
        "collection_errors": sorted(set(collection_errors)),
        "summary_line": summary,
        "exit_code": completed.returncode,
    }


def freeze_baseline() -> dict[str, Any]:
    payload = {
        "stage": "P4P_FAILURE_BASELINE",
        "taken_at": datetime.now().astimezone().isoformat(),
        "why": (
            "tests/methods already fails; a count alone would let a new failure "
            "hide behind an old one, so each node id is stored with a "
            "normalised cause"
        ),
        "gate_after_any_edit": [
            "tests/main_protocol stays fully green",
            "tests/methods gains no new failing node",
            "no existing node's normalised reason degrades",
        ],
        "suites": {
            name: run_suite(name)
            # tests/functional is where run_online_round is actually
            # driven; leaving it out of the baseline once let a real
            # regression through unflagged.
            for name in ("tests/main_protocol", "tests/methods",
                         "tests/functional")
        },
    }
    BASELINE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def compare_baseline() -> dict[str, Any]:
    stored = json.loads(BASELINE.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"suites": {}, "regressed": False}
    for name, before in stored["suites"].items():
        after = run_suite(name)
        old, new = before["failing_nodes"], after["failing_nodes"]
        new_collection = sorted(
            set(after.get("collection_errors") or ())
            - set(before.get("collection_errors") or ()))
        newly = sorted(set(new) - set(old))
        fixed = sorted(set(old) - set(new))
        changed = sorted(
            node for node in set(old) & set(new) if old[node] != new[node]
        )
        regressed = bool(newly or changed or new_collection)
        report["suites"][name] = {
            "before_failing": len(old), "after_failing": len(new),
            "newly_failing": newly, "no_longer_failing": fixed,
            "reason_changed": changed,
            "new_collection_errors": new_collection, "regressed": regressed,
            "summary_line": after["summary_line"],
        }
        report["regressed"] = report["regressed"] or regressed
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("snapshot", "verify", "restore", "baseline", "check"),
    )
    args = parser.parse_args(argv)
    if args.action == "snapshot":
        payload = take_snapshot()
        print("snapshotted %d files -> %s" % (
            payload["file_count"], SNAPSHOT.relative_to(PROJECT_ROOT).as_posix()))
    elif args.action == "verify":
        state = compare_snapshot()
        for key, rows in state.items():
            print("%-8s %d %s" % (key, len(rows), rows[:6]))
    elif args.action == "restore":
        print(restore_snapshot())
    elif args.action == "baseline":
        payload = freeze_baseline()
        for name, suite in payload["suites"].items():
            print("%-22s %s" % (name, suite["summary_line"]))
            print("   failing nodes: %d" % suite["failing_count"])
        print("wrote %s" % BASELINE.relative_to(PROJECT_ROOT).as_posix())
    else:
        report = compare_baseline()
        for name, suite in report["suites"].items():
            print("%-22s before %d -> after %d | new %d | fixed %d | "
                  "reason-changed %d" % (
                      name, suite["before_failing"], suite["after_failing"],
                      len(suite["newly_failing"]), len(suite["no_longer_failing"]),
                      len(suite["reason_changed"])))
            for node in suite["newly_failing"][:10]:
                print("   NEW FAILURE %s" % node)
        print("REGRESSED: %s" % report["regressed"])
        return 1 if report["regressed"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
