from __future__ import annotations

from pathlib import Path

from SelfEvolvingHarnessTS.vnext._canonical import file_sha256
from SelfEvolvingHarnessTS.vnext.preflight import build_data_recovery_manifest
from SelfEvolvingHarnessTS.vnext.recovery import (
    LegacyBundleMatchV1,
    LocalFileMatchV1,
    LocalRecoveryScanV1,
    legacy_bundle_sha,
    quarantine_exact_matches,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registry_station_ids_recover_all_noaa_candidate_paths():
    recovery = build_data_recovery_manifest(ROOT)
    noaa = [row for row in recovery.assets if row.source_id == "noaa_global_hourly"]
    assert len(noaa) == 45
    assert all(row.candidate_paths for row in noaa)
    assert all(any(Path(path).suffix == ".csv" for path in row.candidate_paths) for row in noaa)
    legacy = next(row for row in recovery.assets if row.source_id.startswith("legacy_"))
    assert legacy.actual_sha == recovery.local_legacy_bundle_sha
    assert legacy.status == "PRESENT_VERIFIED"


def test_quarantine_copies_and_rehashes_exact_assets_and_composite_bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "asset.bin"
    asset.write_bytes(b"asset")
    asset_sha = file_sha256(asset)
    meta = source / "monash_clean.meta.jsonl"
    values = source / "monash_clean.npz"
    meta.write_bytes(b"metadata")
    values.write_bytes(b"values")
    bundle_sha = legacy_bundle_sha(source)
    assert bundle_sha is not None

    scan = LocalRecoveryScanV1(
        recovery_manifest_sha="1" * 64,
        search_roots=(str(source),),
        file_matches=(LocalFileMatchV1(
            path=str(asset), basename=asset.name, observed_sha=asset_sha,
            expected_sha=asset_sha, canonical_candidates=("raw/asset.bin",),
            size_bytes=asset.stat().st_size,
        ),),
        legacy_bundle_matches=(LegacyBundleMatchV1(
            directory=str(source), observed_bundle_sha=bundle_sha,
            expected_bundle_sha=bundle_sha, meta_sha=file_sha256(meta),
            values_sha=file_sha256(values),
            size_bytes=meta.stat().st_size + values.stat().st_size,
        ),),
        clean_base_candidates=(), scanned_files=3, skipped_paths=(),
    )
    receipt = quarantine_exact_matches(scan, tmp_path / "quarantine")
    assert len(receipt.copies) == 2
    assert all(row.expected_sha == row.observed_sha for row in receipt.copies)
    assert legacy_bundle_sha(
        tmp_path / "quarantine" / "legacy" / bundle_sha
    ) == bundle_sha
