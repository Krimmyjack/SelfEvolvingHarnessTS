from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            dots = "." * node.level
            found.add(dots + (node.module or ""))
    return found


@pytest.mark.parametrize(
    ("area", "forbidden"),
    [
        ("runtime", {"p6", "harness", "policy", "sandbox", "evaluation"}),
        ("methods", {"p6", "experiments", "evaluation"}),
        ("evaluation", {"p6", "experiments"}),
        ("benchmark", {"p6"}),
    ],
)
def test_active_packages_do_not_import_forbidden_architecture_branches(area, forbidden):
    violations = []
    for path in sorted((ROOT / area).rglob("*.py")):
        for name in _imports(path):
            parts = {part for part in name.lstrip(".").split(".") if part}
            blocked = sorted(parts & forbidden)
            if blocked:
                violations.append(f"{path.relative_to(ROOT)}: {name} -> {blocked}")
    assert violations == []


def test_archived_experiments_are_not_imported_by_active_python():
    violations = []
    for area in ("contracts", "runtime", "methods", "evaluation", "benchmark", "operators"):
        for path in sorted((ROOT / area).rglob("*.py")):
            for name in _imports(path):
                if "experiments" in name.split("."):
                    violations.append(f"{path.relative_to(ROOT)}: {name}")
    assert violations == []
