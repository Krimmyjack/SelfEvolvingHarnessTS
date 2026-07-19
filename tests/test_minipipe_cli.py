from __future__ import annotations

import json

import pytest

from SelfEvolvingHarnessTS.cli.minipipe import (
    _resolve_api_key,
    _safe_genesis_only_resume,
)


def _write_event(path, *, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"event_kind": kind}, sort_keys=True) + "\n")


def test_safe_transport_resume_requires_genesis_only_and_no_cycle(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "private").mkdir(parents=True)
    (run_dir / "private" / "run_context.json").write_text(
        "{}\n", encoding="utf-8", newline="\n"
    )
    _write_event(run_dir / "harness_lineage.jsonl", kind="GENESIS")

    assert _safe_genesis_only_resume(run_dir) is True

    (run_dir / "cycles" / "cycle-000").mkdir(parents=True)
    assert _safe_genesis_only_resume(run_dir) is False


def test_safe_transport_resume_rejects_scientific_lineage_event(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "private").mkdir(parents=True)
    (run_dir / "private" / "run_context.json").write_text(
        "{}\n", encoding="utf-8", newline="\n"
    )
    lineage = run_dir / "harness_lineage.jsonl"
    _write_event(lineage, kind="GENESIS")
    _write_event(lineage, kind="EDIT_EVALUATED")

    assert _safe_genesis_only_resume(run_dir) is False


def test_api_key_file_supports_cross_runtime_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("AGICTO_API_KEY", "environment-secret")
    key_file = tmp_path / "relay.key"
    key_file.write_text("file-secret\n", encoding="utf-8", newline="\n")

    assert _resolve_api_key(key_file) == "file-secret"
    assert _resolve_api_key(None) == "environment-secret"


def test_empty_api_key_source_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("AGICTO_API_KEY", raising=False)
    key_file = tmp_path / "empty.key"
    key_file.write_text("\n", encoding="utf-8", newline="\n")

    with pytest.raises(SystemExit, match="AGICTO_API_KEY or --api-key-file"):
        _resolve_api_key(key_file)
