from __future__ import annotations

from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import parse_json_document
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
    compile_snapshot,
    snapshot_to_dict,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = PACKAGE_ROOT / "artifacts" / "releases" / "m0-h2" / "harness"
EXPECTED_CONTENT_SHA = "8f3845b09322109c878892d88d79810d07d303841574b1df10b3b94e33fca35e"
EXPECTED_RUNTIME_SHA = "7035aef5d57499e21a58b0dc44124255fdfd833eb744815ba63c7befef44f709"


def test_m0_h2_release_is_locked_and_recompilable() -> None:
    snapshot = compile_snapshot(RELEASE_ROOT)

    assert snapshot.harness_content_sha == EXPECTED_CONTENT_SHA
    assert snapshot.runtime_bundle_sha == EXPECTED_RUNTIME_SHA
    assert not snapshot.memories
    assert [
        skill.skill_id for skill in snapshot.skills if skill.skill_kind.value == "capability"
    ] == ["level_shift_contrast_candidate"]


def test_m0_h2_resolved_snapshot_matches_authoring_content() -> None:
    snapshot = compile_snapshot(RELEASE_ROOT)
    resolved = parse_json_document(
        (RELEASE_ROOT / "resolved.snapshot.json").read_bytes()
    )

    assert resolved == snapshot_to_dict(snapshot)
