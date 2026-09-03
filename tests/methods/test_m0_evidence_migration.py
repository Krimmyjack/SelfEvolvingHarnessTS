from __future__ import annotations

from pathlib import Path

from SelfEvolvingHarnessTS.contracts.evidence import EvidenceDisposition
from SelfEvolvingHarnessTS.methods.ttha.m0_evidence_migration import (
    migrate_m0_level_shift_ledger,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = PACKAGE_ROOT / "artifacts" / "releases" / "m0-h2"


def test_m0_level_shift_migrates_read_only_without_causal_overclaim() -> None:
    frozen_before = {
        path.relative_to(RELEASE_DIR): path.read_bytes()
        for path in RELEASE_DIR.rglob("*")
        if path.is_file()
    }

    migrated = migrate_m0_level_shift_ledger(RELEASE_DIR / "capability_ledger.json")

    assert migrated.capability.capability_id == "level_shift_contrast_candidate"
    assert migrated.capability.task_context["data_regime"] == "synthetic"
    assert migrated.capability.consumer_context["scope"] == "forecasting_only"
    assert migrated.evidence.tags == ("synthetic", "forecasting", "legacy_migration")
    assert migrated.evidence.receipt_shas == ()
    assert migrated.evidence.cohort_definition["dataset_membership_available"] is False
    assert "no_dataset_level_causal_claim" in migrated.evidence.causal_scope
    assert (
        migrated.ledger.compile_verdict(migrated.capability.capability_id)
        is EvidenceDisposition.UNRESOLVED
    )
    assert migrated.ledger.read()[0].evidence is migrated.evidence

    assert frozen_before == {
        path.relative_to(RELEASE_DIR): path.read_bytes()
        for path in RELEASE_DIR.rglob("*")
        if path.is_file()
    }
