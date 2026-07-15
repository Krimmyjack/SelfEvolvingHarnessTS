"""Command-line entry points for protocol audit and M0 recovery status."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from ._canonical import canonical_json
from .preflight import (
    EnvironmentLockV1,
    M0ReadinessStatusV1,
    assess_m0_readiness,
    build_data_recovery_manifest,
    default_v02_preflight,
    probe_environment,
    verify_clean_base,
)
from .init_harness import InitCorpusManifestV1, InitHarnessPreregV1
from .protocol import (
    DiscoveryFoldManifestV1,
    HistoricalExposureManifestV1,
    ProtocolResolutionAddendumV1,
    VNextDataUsageManifestV1,
)
from .recovery import (
    build_pinned_download_plan,
    download_pinned_assets,
    promote_quarantined_legacy_bundle,
    promote_verified_clean_base,
    quarantine_exact_matches,
    scan_local_recovery,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(temporary), os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    try:
        payload = (canonical_json(value) + "\n").encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def protocol_audit(root: Path, out: Path | None) -> dict[str, Any]:
    artifacts = {
        "protocol_resolution": ProtocolResolutionAddendumV1(),
        "data_usage": VNextDataUsageManifestV1.from_frozen_v02(root),
        "historical_exposure": HistoricalExposureManifestV1.from_frozen_v02(root),
        "discovery_folds": DiscoveryFoldManifestV1.from_frozen_v02(root),
    }
    if out is not None:
        names = {
            "protocol_resolution": "ProtocolResolutionAddendumV1.json",
            "data_usage": "VNextDataUsageManifestV1.json",
            "historical_exposure": "HistoricalExposureManifestV1.json",
            "discovery_folds": "DiscoveryFoldManifestV1.json",
        }
        for key, artifact in artifacts.items():
            _write(out / names[key], artifact)
    return {
        "status": "PROTOCOL_HARDENING_ARTIFACTS_READY",
        "artifact_shas": {key: value.artifact_sha for key, value in artifacts.items()},
    }


def init_harness_prereg(root: Path, out: Path | None) -> dict[str, Any]:
    corpus = InitCorpusManifestV1.from_frozen_v02(root)
    prereg = InitHarnessPreregV1(init_corpus_sha=corpus.artifact_sha)
    if out is not None:
        _write(out / "InitCorpusManifestV1.json", corpus)
        _write(out / "InitHarnessPreregV1.json", prereg)
    return {
        "status": "INIT_HARNESS_PREREG_READY_H0_NOT_YET_BUILT",
        "init_corpus_sha": corpus.artifact_sha,
        "prereg_sha": prereg.artifact_sha,
        "series_count": len(corpus.members),
        "cohort_counts": dict(corpus.cohort_counts),
        "domain_counts": dict(corpus.domain_counts),
        "forbidden_view_ids": corpus.forbidden_view_ids,
    }
def m0_audit(root: Path, out: Path | None) -> dict[str, Any]:
    integrity = default_v02_preflight(root)
    recovery = build_data_recovery_manifest(root)
    clean = verify_clean_base(root)
    environment_lock = EnvironmentLockV1.from_project(root)
    environment = probe_environment(environment_lock)
    readiness = assess_m0_readiness(recovery, clean, environment)
    erratum_path = root / "results" / "vnext" / "m0" / "ProtocolErratumV1.json"
    if erratum_path.is_file():
        import json

        erratum = json.loads(erratum_path.read_text("utf-8"))
        if str(erratum.get("status", "")).startswith("OPEN_"):
            readiness = M0ReadinessStatusV1(
                data_recovery_sha=recovery.artifact_sha,
                clean_base_sha=clean.artifact_sha,
                environment_probe_sha=environment.artifact_sha,
                status="M0_BLOCKED",
                blockers=tuple((*readiness.blockers, "open_protocol_erratum")),
            )
    if out is not None:
        _write(out / "ArtifactIntegrityReportV1.json", integrity)
        _write(out / "DataRecoveryManifestV1.json", recovery)
        _write(out / "CleanBaseIntegrityManifestV1.json", clean)
        _write(out / "EnvironmentLockV1.json", environment_lock)
        _write(out / "EnvironmentProbeV1.json", environment)
        _write(out / "M0ReadinessStatusV1.json", readiness)
    return {
        "status": readiness.status,
        "task_g_authorized": readiness.task_g_authorized,
        "blockers": readiness.blockers,
        "missing_paths": integrity.missing_paths,
        "recovery_ready": recovery.ready,
        "clean_base_ready": clean.ready,
        "environment_lock_matches": environment.lock_matches,
        "readiness_sha": readiness.artifact_sha,
    }


def recovery_scan(
    root: Path,
    search_roots: list[Path],
    out: Path | None,
    quarantine: Path | None,
) -> dict[str, Any]:
    scan = scan_local_recovery(root, search_roots)
    receipt = quarantine_exact_matches(scan, quarantine) if quarantine is not None else None
    if out is not None:
        _write(out / "LocalRecoveryScanV1.json", scan)
        if receipt is not None:
            _write(out / "QuarantineReceiptV1.json", receipt)
    return {
        "status": "LOCAL_RECOVERY_SCAN_COMPLETE",
        "scan_sha": scan.artifact_sha,
        "scanned_files": scan.scanned_files,
        "exact_file_matches": len(scan.file_matches),
        "exact_legacy_bundle_matches": len(scan.legacy_bundle_matches),
        "clean_base_candidates": len(scan.clean_base_candidates),
        "ready_clean_base_candidates": sum(row.ready for row in scan.clean_base_candidates),
        "quarantine_receipt_sha": receipt.artifact_sha if receipt is not None else None,
        "quarantine_copies": len(receipt.copies) if receipt is not None else 0,
    }


def pinned_recovery(
    root: Path,
    quarantine: Path,
    out: Path | None,
) -> dict[str, Any]:
    receipt = download_pinned_assets(root, quarantine)
    if out is not None:
        _write(out / "PinnedDownloadReceiptV1.json", receipt)
    counts: dict[str, int] = {}
    for row in receipt.results:
        counts[row.status] = counts.get(row.status, 0) + 1
    return {
        "status": "PINNED_RECOVERY_COMPLETE",
        "receipt_sha": receipt.artifact_sha,
        "result_counts": counts,
        "manual_only_sources": receipt.manual_only_sources,
    }


def recovery_plan(root: Path, out: Path | None) -> dict[str, Any]:
    plan = build_pinned_download_plan(root)
    if out is not None:
        _write(out / "PinnedDownloadPlanV1.json", plan)
    return {
        "status": "PINNED_RECOVERY_PLAN_READY",
        "plan_sha": plan.artifact_sha,
        "automatic_assets": len(plan.specs),
        "manual_only_sources": plan.manual_only_sources,
    }


def legacy_promote(
    root: Path,
    quarantine: Path,
    out: Path | None,
) -> dict[str, Any]:
    receipt = promote_quarantined_legacy_bundle(root, quarantine)
    if out is not None:
        _write(out / "LegacyPromotionReceiptV1.json", receipt)
    return {
        "status": "LEGACY_BUNDLE_PROMOTED",
        "receipt_sha": receipt.artifact_sha,
        "previous_bundle_sha": receipt.previous_bundle_sha,
        "promoted_bundle_sha": receipt.promoted_bundle_sha,
    }


def clean_base_promote(
    root: Path,
    source: Path,
    out: Path | None,
) -> dict[str, Any]:
    receipt = promote_verified_clean_base(root, source)
    if out is not None:
        _write(out / "CleanBasePromotionReceiptV1.json", receipt)
    return {
        "status": "CLEAN_BASE_PROMOTED",
        "receipt_sha": receipt.artifact_sha,
        "verified_records": receipt.verified_records,
        "content_sha": receipt.destination_content_sha,
        "destination": receipt.destination_root,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsharness-vnext")
    parser.add_argument(
        "command", choices=(
            "protocol-audit", "m0-audit", "recovery-scan", "pinned-recovery",
            "legacy-promote", "recovery-plan",
            "init-harness-prereg",
            "clean-base-promote",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--search-root", action="append", type=Path, default=[])
    parser.add_argument("--quarantine", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args(argv)
    if args.command == "protocol-audit":
        result = protocol_audit(args.root, args.out)
    elif args.command == "init-harness-prereg":
        result = init_harness_prereg(args.root, args.out)
    elif args.command == "m0-audit":
        result = m0_audit(args.root, args.out)
    elif args.command == "recovery-plan":
        result = recovery_plan(args.root, args.out)
    elif args.command == "recovery-scan":
        if not args.search_root:
            parser.error("recovery-scan requires at least one --search-root")
        result = recovery_scan(
            args.root, args.search_root, args.out, args.quarantine,
        )
    elif args.command == "pinned-recovery":
        if args.quarantine is None:
            parser.error("pinned-recovery requires --quarantine")
        result = pinned_recovery(args.root, args.quarantine, args.out)
    elif args.command == "clean-base-promote":
        if args.source is None:
            parser.error("clean-base-promote requires --source")
        result = clean_base_promote(args.root, args.source, args.out)
    else:
        if args.quarantine is None:
            parser.error("legacy-promote requires --quarantine")
        result = legacy_promote(args.root, args.quarantine, args.out)
    print(canonical_json(result))
    return 0 if result["status"] != "M0_BLOCKED" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
