from __future__ import annotations

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

