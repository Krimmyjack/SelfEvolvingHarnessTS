from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_AREAS = ("contracts", "operators", "conditioning", "runtime", "methods", "evaluation")
REMOVED_TOP_LEVEL = {
    "benchmark",
    "config",
    "diagnostics",
    "evaluators",
    "fast_path",
    "harness",
    "llm",
    "memory",
    "models",
    "p6",
    "policy",
    "sandbox",
    "slow_path",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add("." * node.level + (node.module or ""))
    return found


def _top_level(path: Path, name: str) -> str:
    if name.startswith("."):
        level = len(name) - len(name.lstrip("."))
        module_parts = [part for part in name.lstrip(".").split(".") if part]
        package = list(path.relative_to(ROOT).with_suffix("").parts[:-1])
        keep = max(0, len(package) - (level - 1))
        resolved = package[:keep] + module_parts
    else:
        resolved = [part for part in name.split(".") if part]
    if resolved and resolved[0] == ROOT.name:
        resolved = resolved[1:]
    return resolved[0] if resolved else ""


def test_retired_top_level_namespaces_are_absent():
    assert sorted(name for name in REMOVED_TOP_LEVEL if (ROOT / name).exists()) == []


def test_active_code_does_not_import_retired_namespaces():
    violations = []
    for area in ACTIVE_AREAS:
        for path in sorted((ROOT / area).rglob("*.py")):
            for name in _imports(path):
                top_level = _top_level(path, name)
                if top_level in REMOVED_TOP_LEVEL:
                    violations.append(f"{path.relative_to(ROOT)}: {name} -> {top_level}")
    assert violations == []


def test_historical_experiments_are_not_importable_source():
    python_files = sorted((ROOT / "experiments").rglob("*.py"))
    assert python_files == []
