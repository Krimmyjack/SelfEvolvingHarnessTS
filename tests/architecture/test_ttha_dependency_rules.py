from __future__ import annotations

import ast
import json
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.public_boundary import FORBIDDEN_PUBLIC_KEYS


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append("." * node.level + (node.module or ""))
    return tuple(found)


def test_ttha_never_imports_evaluation_or_h_ref():
    violations = []
    for path in sorted((ROOT / "methods/ttha").rglob("*.py")):
        for name in _imports(path):
            if "evaluation" in name or "h_ref_v02" in name:
                violations.append(f"{path.relative_to(ROOT)}: {name}")
    assert violations == []


def test_generic_runtime_has_no_active_method_imports_except_retirement_fossil():
    violations = []
    for path in sorted((ROOT / "runtime").rglob("*.py")):
        for name in _imports(path):
            if "methods.ttha" in name or "methods.h_ref_v02" in name:
                violations.append((path.relative_to(ROOT).as_posix(), name))
    # Task 14 moves this final benchmark-v0.2 compatibility implementation out
    # of generic runtime and tightens this assertion to an empty list.
    assert violations == [("runtime/fast_path.py", "..methods.h_ref_v02.config")]


def test_prompt_and_request_constructors_do_not_read_files():
    for relative in ("methods/ttha/agent_core.py", "runtime/agent_backend.py"):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name) and function.id == "open":
                forbidden.append("open")
            if isinstance(function, ast.Attribute) and function.attr.startswith("read_"):
                forbidden.append(function.attr)
        assert forbidden == [], relative
        assert "private/" not in path.read_text(encoding="utf-8").lower()


def test_ttha_transitive_source_has_no_private_baseline_reference():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "methods/ttha").rglob("*.py"))
    )
    assert "evaluation.minipipe" not in text
    assert "fixed_program_baseline" not in text
    assert "evaluation/minipipe/baselines" not in text


def test_public_schema_properties_are_disjoint_from_forbidden_keys():
    for filename in ("failure_pattern_card_v1.json", "public_case_view_v1.json"):
        schema = json.loads((ROOT / "evaluation/minipipe/schemas" / filename).read_text())
        properties = set(schema["properties"])
        assert properties.isdisjoint(FORBIDDEN_PUBLIC_KEYS)


def test_wall_substrate_is_read_only_not_an_edit_surface():
    surfaces = json.loads(
        (ROOT / "methods/ttha/harness/harness_surfaces.json").read_text()
    )
    editable = json.dumps(surfaces["surfaces"], sort_keys=True)
    for forbidden in (
        "observable_feature_v1.json",
        "contracts/observables.py",
        "probes/features.py",
        "public_tools.py",
    ):
        assert forbidden not in editable
    read_only = json.dumps(surfaces["read_only"], sort_keys=True)
    assert "contracts/observables.py" in read_only
    assert "observable_feature_v1.json" in read_only


def test_minipipe_and_ttha_do_not_import_h_ref():
    violations = []
    roots = (ROOT / "methods/ttha", ROOT / "evaluation/minipipe")
    for source_root in roots:
        for path in sorted(source_root.rglob("*.py")):
            for name in _imports(path):
                if "h_ref_v02" in name or "_frozen_reference" in name:
                    violations.append(f"{path.relative_to(ROOT)}: {name}")
    assert violations == []
