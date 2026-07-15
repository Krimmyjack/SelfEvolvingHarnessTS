from __future__ import annotations

from dataclasses import replace

import pytest

from SelfEvolvingHarnessTS.vnext.access import (
    OneShotAccessControllerV1,
    OneShotAccessError,
    OneShotAccessManifestV1,
)


def _sha(character: str) -> str:
    return character * 64


def _manifest(resource: str = "sa_validation") -> OneShotAccessManifestV1:
    return OneShotAccessManifestV1(
        resource_kind=resource,
        campaign_id=f"{resource}-campaign",
        prereg_sha=_sha("1"), authorization_sha=_sha("2"),
        resource_manifest_sha=_sha("3"), materialization_sha=_sha("4"),
        method_sha=_sha("5"), roster_sha=_sha("6"), runner_code_sha=_sha("7"),
        environment_sha=_sha("8"), budget_sha=_sha("9"), seed_book_sha=_sha("a"),
        initial_checkpoint_sha=_sha("b"),
    )


def test_loader_runs_only_after_durable_access_reservation(tmp_path):
    manifest = _manifest()
    seen = []
    with OneShotAccessControllerV1(tmp_path, manifest) as controller:
        controller.authorize(manifest.authorization_sha)
        receipt = controller.reserve("run-1")

        def loader():
            text = controller.path.read_text("utf-8")
            assert '"event":"access_reserved"' in text
            seen.append(receipt.receipt_sha)
            return "aggregate"

        assert controller.load(receipt, loader) == "aggregate"
        terminal = controller.record_terminal(receipt, "passed", _sha("c"))
        assert terminal.resource_kind == "sa_validation"
    assert seen == [receipt.receipt_sha]

    with OneShotAccessControllerV1(tmp_path, manifest) as replayed:
        assert replayed.terminal_artifact == terminal
        with pytest.raises(OneShotAccessError, match="already has a reservation"):
            replayed.reserve("run-2")


def test_pending_reservation_resumes_only_with_exact_binding(tmp_path):
    manifest = _manifest("support_b")
    controller = OneShotAccessControllerV1(tmp_path, manifest)
    controller.authorize(manifest.authorization_sha)
    receipt = controller.reserve("run-1")
    controller.checkpoint(receipt, _sha("c"))
    controller.interrupt_infrastructure(receipt, _sha("c"))
    controller.close()

    with OneShotAccessControllerV1(tmp_path, manifest) as replayed:
        binding = replayed.expected_resume_binding()
        with pytest.raises(OneShotAccessError, match="differs"):
            replayed.resume_exact(replace(binding, environment_sha=_sha("d")))
        assert replayed.resume_exact(binding) == receipt
        replayed.record_terminal(receipt, "failed_gate", _sha("e"))


def test_same_resource_cannot_open_second_manifest_or_concurrent_ledger(tmp_path):
    manifest = _manifest()
    first = OneShotAccessControllerV1(tmp_path, manifest)
    try:
        with pytest.raises(OneShotAccessError, match="locked"):
            OneShotAccessControllerV1(tmp_path, manifest)
    finally:
        first.close()
    changed = replace(manifest, campaign_id="another-campaign")
    with pytest.raises(OneShotAccessError, match="another manifest"):
        OneShotAccessControllerV1(tmp_path, changed)


@pytest.mark.parametrize("crash_point", [
    "after_reservation", "after_running", "after_first_series",
    "after_first_dataset", "after_terminal_before_lifecycle_close",
])
def test_crash_points_never_create_a_second_access_opportunity(tmp_path, crash_point):
    manifest = _manifest("support_b")
    controller = OneShotAccessControllerV1(tmp_path, manifest)
    controller.authorize(manifest.authorization_sha)
    receipt = controller.reserve("run-1")
    if crash_point != "after_reservation":
        controller.load(receipt, lambda: None)
    if crash_point == "after_first_series":
        controller.checkpoint(receipt, _sha("c"))
    if crash_point == "after_first_dataset":
        controller.checkpoint(receipt, _sha("d"))
    if crash_point == "after_terminal_before_lifecycle_close":
        controller.record_terminal(receipt, "passed", _sha("e"))
    controller.close()

    with OneShotAccessControllerV1(tmp_path, manifest) as replayed:
        with pytest.raises(OneShotAccessError, match="already has a reservation"):
            replayed.reserve("run-2")
        if crash_point == "after_terminal_before_lifecycle_close":
            assert replayed.pending_reservation() is None
        else:
            assert replayed.resume_exact(replayed.expected_resume_binding()) == receipt


def test_torn_or_tampered_access_ledger_fails_loud(tmp_path):
    manifest = _manifest()
    controller = OneShotAccessControllerV1(tmp_path, manifest)
    path = controller.path
    controller.close()
    original = path.read_text("utf-8")
    path.write_text(original.replace('"event":"resource_freeze"', '"event":"authorize"'), "utf-8")
    with pytest.raises(OneShotAccessError, match="hash"):
        OneShotAccessControllerV1(tmp_path, manifest)
