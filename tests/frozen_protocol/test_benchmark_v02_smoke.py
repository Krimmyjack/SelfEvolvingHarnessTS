from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.evaluation import benchmark_v02
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.__main__ import build_parser, main
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.programs import apply_program


MODULES = (
    "aggregate",
    "baselines",
    "corruption",
    "datasets",
    "dev_eval",
    "ingestion",
    "ledger",
    "materialize",
    "method_api",
    "method_compat",
    "metrics",
    "models",
    "power",
    "probe",
    "programs",
    "registry",
    "report",
    "runner",
    "sources",
    "spatial",
    "split",
    "trainers",
    "workspace",
)


def test_benchmark_has_one_importable_package_owner():
    prefix = "SelfEvolvingHarnessTS.evaluation.benchmark_v02"
    for module in MODULES:
        assert importlib.import_module(f"{prefix}.{module}") is not None


def test_package_cli_exposes_the_frozen_phases_and_dispatches(tmp_path):
    commands = build_parser()._subparsers._group_actions[0].choices
    assert set(commands) == {
        "acquire",
        "probe",
        "freeze",
        "dry-run",
        "confirm",
        "dev-eval",
        "campaign-freeze",
        "final-eval",
    }
    calls = []
    handlers = {"probe": lambda args: calls.append((args.root, args.out)) or 0}
    assert main(
        ["probe", "--root", str(tmp_path / "data"), "--out", str(tmp_path / "out")],
        handlers=handlers,
    ) == 0
    assert calls == [(str(tmp_path / "data"), str(tmp_path / "out"))]


def test_benchmark_bundled_inputs_and_program_smoke():
    data = Path(benchmark_v02.__file__).resolve().parent / "data"
    assert (data / "acquisition_manifest.json").is_file()
    assert (data / "legacy" / "monash_clean.meta.jsonl").is_file()
    assert (data / "legacy" / "monash_clean.npz").is_file()
    assert (data / "legacy" / "u_admission_v2_traffic_hourly.json").is_file()
    values = np.array([1.0, np.nan, 3.0, 4.0])
    raw = apply_program("raw", values, period=2)
    assert np.array_equal(raw, values, equal_nan=True)
