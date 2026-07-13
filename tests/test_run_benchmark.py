from __future__ import annotations

import json

from SelfEvolvingHarnessTS.run_benchmark import build_parser, main


def test_cli_exposes_all_frozen_phases():
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices
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


def test_cli_dispatches_without_embedding_phase_logic(tmp_path):
    calls = []
    handlers = {"probe": lambda args: calls.append((args.command, args.root, args.out)) or 0}
    code = main(
        ["probe", "--root", str(tmp_path / "data"), "--out", str(tmp_path / "out")],
        handlers=handlers,
    )
    assert code == 0
    assert calls == [("probe", str(tmp_path / "data"), str(tmp_path / "out"))]


def test_acquire_manual_status_writes_account_gated_manifest(tmp_path):
    root = tmp_path / "data"
    code = main(["acquire", "--root", str(root), "--manual-status"])
    assert code == 0
    payload = json.loads((root / "acquisition_manifest.json").read_text("utf-8"))
    by_id = {row["source_id"]: row for row in payload["results"]}
    assert by_id["entsoe_transparency"]["status"] == "manual_required"
    assert by_id["gefcom2012"]["status"] == "manual_required"
    assert by_id["gefcom2014"]["status"] == "manual_required"
    assert by_id["monash_hf"]["status"] == "automatic_skipped"


def test_default_probe_and_freeze_handlers_call_workspace(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "SelfEvolvingHarnessTS.benchmark.workspace.probe_workspace",
        lambda root, out: calls.append(("probe", root, out)),
    )
    monkeypatch.setattr(
        "SelfEvolvingHarnessTS.benchmark.workspace.freeze_workspace",
        lambda root, out: calls.append(("freeze", root, out)),
    )
    root, out = str(tmp_path / "data"), str(tmp_path / "out")
    assert main(["probe", "--root", root, "--out", out]) == 0
    assert main(["freeze", "--root", root, "--out", out]) == 0
    assert calls == [("probe", root, out), ("freeze", root, out)]


def test_default_dev_handler_requires_frozen_baseline_roster(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "SelfEvolvingHarnessTS.benchmark.dev_eval.run_dev_evaluation",
        lambda root, out: calls.append((root, out)),
    )
    root, out = str(tmp_path / "data"), str(tmp_path / "out")
    roster = "raw,best_fixed,h_ref,oracle_transfer,oracle_insample"
    assert main(["dev-eval", "--root", root, "--out", out, "--baselines", roster]) == 0
    assert calls == [(root, out)]
