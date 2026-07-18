from __future__ import annotations

import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append("." * node.level + (node.module or ""))
    return tuple(values)


def test_no_active_h_ref_method_or_runtime_remains():
    assert not (ROOT / "methods" / "h_ref_v02").exists()
    assert not (ROOT / "runtime" / "fast_path.py").exists()
    methods = importlib.import_module("SelfEvolvingHarnessTS.methods")
    assert not hasattr(methods, "HRefV02Method")


def test_frozen_reference_imports_are_benchmark_private_only():
    violations = []
    for source_root in (
        ROOT / "contracts",
        ROOT / "operators",
        ROOT / "runtime",
        ROOT / "methods",
        ROOT / "evaluation" / "minipipe",
        ROOT / "cli",
    ):
        for path in sorted(source_root.rglob("*.py")):
            for name in _imports(path):
                if "_frozen_reference" in name or "h_ref_v02" in name:
                    violations.append(f"{path.relative_to(ROOT)}: {name}")
    assert violations == []


def test_benchmark_is_the_only_owner_of_frozen_reference_imports():
    benchmark = ROOT / "evaluation" / "benchmark_v02"
    package_root = benchmark / "_frozen_reference"
    assert package_root.is_dir()
    public_init = (benchmark / "__init__.py").read_text(encoding="utf-8")
    assert "run_legacy_reference_batch" not in public_init
    assert "LegacyReferenceState" not in public_init
