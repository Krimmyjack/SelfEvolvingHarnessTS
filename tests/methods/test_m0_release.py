from __future__ import annotations

from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import parse_json_document
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
    compile_compatible_snapshot,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = PACKAGE_ROOT / "artifacts" / "releases" / "m0-h2"
RELEASE_ROOT = RELEASE_DIR / "harness"
F1_BINDING = (
    PACKAGE_ROOT / "artifacts" / "compatibility" / "f1-h2" / "runtime_binding.json"
)
EXPECTED_CONTENT_SHA = "8f3845b09322109c878892d88d79810d07d303841574b1df10b3b94e33fca35e"
EXPECTED_RUNTIME_SHA = "7035aef5d57499e21a58b0dc44124255fdfd833eb744815ba63c7befef44f709"


def test_m0_h2_release_keeps_historical_lock_and_recompiles_compatibly() -> None:
    historical_lock = parse_json_document(
        (RELEASE_ROOT / "snapshot.lock.json").read_bytes()
    )
    snapshot = compile_compatible_snapshot(
        RELEASE_ROOT,
        expected_harness_content_sha=EXPECTED_CONTENT_SHA,
    )

    assert snapshot.harness_content_sha == EXPECTED_CONTENT_SHA
    assert historical_lock["harness_content_sha"] == EXPECTED_CONTENT_SHA
    assert historical_lock["runtime_bundle_sha"] == EXPECTED_RUNTIME_SHA
    assert snapshot.runtime_bundle_sha != EXPECTED_RUNTIME_SHA
    assert not snapshot.memories
    assert [
        skill.skill_id for skill in snapshot.skills if skill.skill_kind.value == "capability"
    ] == ["level_shift_contrast_candidate"]


def test_m0_h2_resolved_snapshot_remains_the_historical_release() -> None:
    resolved = parse_json_document(
        (RELEASE_ROOT / "resolved.snapshot.json").read_bytes()
    )

    assert resolved["harness_content_sha"] == EXPECTED_CONTENT_SHA
    assert resolved["runtime_bundle_sha"] == EXPECTED_RUNTIME_SHA


def test_m0_release_manifest_and_capability_ledger_match_snapshot() -> None:
    snapshot = compile_compatible_snapshot(
        RELEASE_ROOT,
        expected_harness_content_sha=EXPECTED_CONTENT_SHA,
    )
    historical_lock = parse_json_document(
        (RELEASE_ROOT / "snapshot.lock.json").read_bytes()
    )
    manifest = parse_json_document((RELEASE_DIR / "release_manifest.json").read_bytes())
    ledger = parse_json_document((RELEASE_DIR / "capability_ledger.json").read_bytes())
    receipt = parse_json_document((RELEASE_DIR / "restore_receipt.json").read_bytes())

    assert manifest["implementation_commit"] == "b2b799dbf352b564551b8706a2366cfac685f980"
    assert manifest["harness"]["harness_content_sha"] == snapshot.harness_content_sha
    assert manifest["harness"]["runtime_bundle_sha"] == historical_lock["runtime_bundle_sha"]
    assert receipt["status"] == "PASS"
    assert receipt["archive"]["sha256"] == manifest["private_bundle"]["archive_sha256"]

    capability = ledger["capabilities"][0]
    assert capability["capability_id"] == "level_shift_contrast_candidate"
    assert capability["heldout_reuse"]["positive_incremental_reuse"] == 3
    assert capability["heldout_reuse"]["out_of_scope_behavior_changes"] == 0
    assert capability["heldout_reuse"]["out_of_scope_case_count"] == 44


def test_f1_runtime_binding_reuses_h2_content_without_claiming_an_edit() -> None:
    binding = parse_json_document(F1_BINDING.read_bytes())
    snapshot = compile_compatible_snapshot(
        RELEASE_ROOT,
        expected_harness_content_sha=EXPECTED_CONTENT_SHA,
    )

    assert binding["harness_edit"] is False
    assert binding["content_identity_verified"] is True
    assert binding["harness_content_sha"] == snapshot.harness_content_sha
    assert binding["historical_m0_runtime_bundle_sha"] == EXPECTED_RUNTIME_SHA
    assert binding["f1_runtime_bundle_sha"] == snapshot.runtime_bundle_sha
