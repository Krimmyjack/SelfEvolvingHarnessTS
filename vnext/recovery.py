"""SHA-first local recovery for frozen benchmark-v0.2 inputs.

Discovery never promotes a candidate into a canonical data path.  Exact matches may be
copied into a project-local quarantine, where their bytes are hashed again and recorded
in an immutable receipt.  Promotion/rebuilding is a later, explicit M0 operation.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ._canonical import file_sha256, require_sha, sha256
from .preflight import (
    CleanBaseIntegrityManifestV1,
    build_data_recovery_manifest,
    verify_clean_base_at,
)


LEGACY_META = "monash_clean.meta.jsonl"
LEGACY_VALUES = "monash_clean.npz"


def legacy_bundle_sha(directory: Path) -> str | None:
    meta = directory / LEGACY_META
    values = directory / LEGACY_VALUES
    if not meta.is_file() or not values.is_file():
        return None
    metadata_bytes = meta.read_bytes()
    values_bytes = values.read_bytes()
    payload = (
        b"benchmark-legacy-bundle-v1\0"
        + len(metadata_bytes).to_bytes(8, "big")
        + metadata_bytes
        + len(values_bytes).to_bytes(8, "big")
        + values_bytes
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LocalFileMatchV1:
    path: str
    basename: str
    observed_sha: str
    expected_sha: str
    canonical_candidates: tuple[str, ...]
    size_bytes: int

    def __post_init__(self) -> None:
        require_sha(self.observed_sha, "observed_sha")
        require_sha(self.expected_sha, "expected_sha")
        if self.observed_sha != self.expected_sha:
            raise ValueError("a recovery match must be byte-exact")


@dataclass(frozen=True)
class LegacyBundleMatchV1:
    directory: str
    observed_bundle_sha: str
    expected_bundle_sha: str
    meta_sha: str
    values_sha: str
    size_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "observed_bundle_sha", "expected_bundle_sha", "meta_sha", "values_sha",
        ):
            require_sha(getattr(self, name), name)
        if self.observed_bundle_sha != self.expected_bundle_sha:
            raise ValueError("a legacy recovery match must be byte-exact")


@dataclass(frozen=True)
class CleanBaseCandidateV1:
    path: str
    integrity_sha: str
    verified_records: int
    expected_records: int
    ready: bool

    def __post_init__(self) -> None:
        require_sha(self.integrity_sha, "integrity_sha")


@dataclass(frozen=True)
class LocalRecoveryScanV1:
    recovery_manifest_sha: str
    search_roots: tuple[str, ...]
    file_matches: tuple[LocalFileMatchV1, ...]
    legacy_bundle_matches: tuple[LegacyBundleMatchV1, ...]
    clean_base_candidates: tuple[CleanBaseCandidateV1, ...]
    scanned_files: int
    skipped_paths: tuple[str, ...]
    schema_version: str = "vnext-local-recovery-scan/1"

    def __post_init__(self) -> None:
        require_sha(self.recovery_manifest_sha, "recovery_manifest_sha")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class QuarantineCopyV1:
    source: str
    destination: str
    expected_sha: str
    observed_sha: str
    kind: str

    def __post_init__(self) -> None:
        require_sha(self.expected_sha, "expected_sha")
        require_sha(self.observed_sha, "observed_sha")
        if self.expected_sha != self.observed_sha:
            raise ValueError("quarantine copy changed bytes")
        if self.kind not in {"asset", "legacy_bundle"}:
            raise ValueError("unknown quarantine copy kind")


@dataclass(frozen=True)
class QuarantineReceiptV1:
    scan_sha: str
    quarantine_root: str
    copies: tuple[QuarantineCopyV1, ...]
    schema_version: str = "vnext-quarantine-receipt/1"

    def __post_init__(self) -> None:
        require_sha(self.scan_sha, "scan_sha")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class PinnedDownloadSpecV1:
    source_id: str
    url: str
    basename: str
    expected_sha: str
    canonical_candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha(self.expected_sha, "expected_sha")
        if not self.url.startswith("https://"):
            raise ValueError("recovery downloads require HTTPS")


@dataclass(frozen=True)
class PinnedDownloadResultV1:
    source_id: str
    url: str
    expected_sha: str
    observed_sha: str | None
    size_bytes: int | None
    status: str
    destination: str | None
    error: str | None = None

    def __post_init__(self) -> None:
        require_sha(self.expected_sha, "expected_sha")
        if self.observed_sha is not None:
            require_sha(self.observed_sha, "observed_sha")
        if self.status not in {"EXACT_QUARANTINED", "SHA_MISMATCH", "DOWNLOAD_ERROR"}:
            raise ValueError("unknown pinned download status")


@dataclass(frozen=True)
class PinnedDownloadPlanV1:
    recovery_manifest_sha: str
    specs: tuple[PinnedDownloadSpecV1, ...]
    manual_only_sources: tuple[str, ...]
    schema_version: str = "vnext-pinned-download-plan/1"

    def __post_init__(self) -> None:
        require_sha(self.recovery_manifest_sha, "recovery_manifest_sha")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class PinnedDownloadReceiptV1:
    recovery_manifest_sha: str
    results: tuple[PinnedDownloadResultV1, ...]
    manual_only_sources: tuple[str, ...]
    schema_version: str = "vnext-pinned-download-receipt/1"

    def __post_init__(self) -> None:
        require_sha(self.recovery_manifest_sha, "recovery_manifest_sha")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class LegacyPromotionReceiptV1:
    expected_bundle_sha: str
    previous_bundle_sha: str | None
    promoted_bundle_sha: str
    source_directory: str
    destination_directory: str
    displaced_backup_directory: str | None
    schema_version: str = "vnext-legacy-promotion-receipt/1"

    def __post_init__(self) -> None:
        for name in ("expected_bundle_sha", "promoted_bundle_sha"):
            require_sha(getattr(self, name), name)
        if self.previous_bundle_sha is not None:
            require_sha(self.previous_bundle_sha, "previous_bundle_sha")
        if self.promoted_bundle_sha != self.expected_bundle_sha:
            raise ValueError("legacy promotion did not restore the frozen bundle")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


def clean_base_content_sha(manifest: CleanBaseIntegrityManifestV1) -> str:
    """Path-independent binding for a complete clean-base verification result."""
    return sha256({
        "registry_sha": manifest.registry_sha,
        "records": manifest.records,
        "expected_record_count": manifest.expected_record_count,
    })


@dataclass(frozen=True)
class CleanBasePromotionReceiptV1:
    source_root: str
    destination_root: str
    registry_sha: str
    source_content_sha: str
    destination_content_sha: str
    verified_records: int
    promotion_mode: str = "verified_copy_then_atomic_directory_replace"
    schema_version: str = "vnext-clean-base-promotion-receipt/1"

    def __post_init__(self) -> None:
        for name in ("registry_sha", "source_content_sha", "destination_content_sha"):
            require_sha(getattr(self, name), name)
        if self.source_content_sha != self.destination_content_sha:
            raise ValueError("clean-base promotion changed verified content")
        if self.verified_records != 1919:
            raise ValueError("clean-base promotion must cover all 1919 records")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


def _walk_files(search_root: Path) -> Iterable[Path]:
    for directory, names, files in os.walk(search_root, followlinks=False):
        names[:] = [name for name in names if name not in {".git", "__pycache__"}]
        base = Path(directory)
        for name in files:
            yield base / name


def scan_local_recovery(
    project_root: Path | str,
    search_roots: Sequence[Path | str],
) -> LocalRecoveryScanV1:
    project = Path(project_root).resolve()
    recovery = build_data_recovery_manifest(project)
    expected_by_name: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for asset in recovery.assets:
        for candidate in asset.candidate_paths:
            expected_by_name.setdefault(Path(candidate).name, []).append(
                (asset.expected_sha, asset.candidate_paths)
            )

    roots = tuple(Path(root).expanduser().resolve() for root in search_roots)
    file_matches: list[LocalFileMatchV1] = []
    legacy_dirs: set[Path] = set()
    clean_dirs: set[Path] = set()
    skipped: list[str] = []
    scanned = 0
    for root in roots:
        if not root.exists():
            skipped.append(f"missing:{root}")
            continue
        try:
            for path in _walk_files(root):
                scanned += 1
                if path.name in {LEGACY_META, LEGACY_VALUES}:
                    legacy_dirs.add(path.parent)
                if path.parent.name == "clean_base":
                    clean_dirs.add(path.parent)
                expectations = expected_by_name.get(path.name, ())
                if not expectations:
                    continue
                try:
                    observed = file_sha256(path)
                except OSError as exc:
                    skipped.append(f"unreadable:{path}:{type(exc).__name__}")
                    continue
                for expected, candidates in expectations:
                    if observed == expected:
                        file_matches.append(LocalFileMatchV1(
                            path=str(path), basename=path.name,
                            observed_sha=observed, expected_sha=expected,
                            canonical_candidates=tuple(candidates),
                            size_bytes=path.stat().st_size,
                        ))
        except OSError as exc:
            skipped.append(f"unreadable_root:{root}:{type(exc).__name__}")
        try:
            clean_dirs.update(
                path for path in root.rglob("clean_base") if path.is_dir()
            )
        except OSError as exc:
            skipped.append(f"clean_base_scan:{root}:{type(exc).__name__}")

    legacy_matches: list[LegacyBundleMatchV1] = []
    expected_legacy = recovery.legacy_registry_binding_sha
    if expected_legacy is not None:
        for directory in sorted(legacy_dirs):
            observed = legacy_bundle_sha(directory)
            if observed != expected_legacy:
                continue
            meta = directory / LEGACY_META
            values = directory / LEGACY_VALUES
            legacy_matches.append(LegacyBundleMatchV1(
                directory=str(directory), observed_bundle_sha=observed,
                expected_bundle_sha=expected_legacy,
                meta_sha=file_sha256(meta), values_sha=file_sha256(values),
                size_bytes=meta.stat().st_size + values.stat().st_size,
            ))

    clean_candidates: list[CleanBaseCandidateV1] = []
    for directory in sorted(clean_dirs):
        integrity = verify_clean_base_at(project, directory)
        verified = sum(row.status == "PRESENT_VERIFIED" for row in integrity.records)
        clean_candidates.append(CleanBaseCandidateV1(
            path=str(directory), integrity_sha=integrity.artifact_sha,
            verified_records=verified, expected_records=integrity.expected_record_count,
            ready=integrity.ready,
        ))

    return LocalRecoveryScanV1(
        recovery_manifest_sha=recovery.artifact_sha,
        search_roots=tuple(str(root) for root in roots),
        file_matches=tuple(sorted(file_matches, key=lambda row: (row.expected_sha, row.path))),
        legacy_bundle_matches=tuple(sorted(legacy_matches, key=lambda row: row.directory)),
        clean_base_candidates=tuple(sorted(clean_candidates, key=lambda row: row.path)),
        scanned_files=scanned,
        skipped_paths=tuple(sorted(skipped)),
    )


def _copy_exact(source: Path, destination: Path, expected_sha: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != expected_sha:
            raise FileExistsError(f"quarantine collision: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if file_sha256(temporary) != expected_sha:
        temporary.unlink()
        raise IOError(f"copy verification failed: {source}")
    os.replace(temporary, destination)


def _replace_exact(source: Path, destination: Path, expected_sha: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".promote.tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if file_sha256(temporary) != expected_sha:
        temporary.unlink()
        raise IOError(f"promotion copy verification failed: {source}")
    os.replace(temporary, destination)


def quarantine_exact_matches(
    scan: LocalRecoveryScanV1,
    quarantine_root: Path | str,
) -> QuarantineReceiptV1:
    quarantine = Path(quarantine_root).resolve()
    copies: list[QuarantineCopyV1] = []
    for match in scan.file_matches:
        destination = quarantine / "assets" / match.expected_sha / match.basename
        _copy_exact(Path(match.path), destination, match.expected_sha)
        copies.append(QuarantineCopyV1(
            source=match.path, destination=str(destination),
            expected_sha=match.expected_sha, observed_sha=file_sha256(destination),
            kind="asset",
        ))
    for match in scan.legacy_bundle_matches:
        source_dir = Path(match.directory)
        destination_dir = quarantine / "legacy" / match.expected_bundle_sha
        _copy_exact(source_dir / LEGACY_META, destination_dir / LEGACY_META, match.meta_sha)
        _copy_exact(source_dir / LEGACY_VALUES, destination_dir / LEGACY_VALUES, match.values_sha)
        observed = legacy_bundle_sha(destination_dir)
        if observed != match.expected_bundle_sha:
            raise IOError("quarantined legacy bundle failed composite verification")
        copies.append(QuarantineCopyV1(
            source=match.directory, destination=str(destination_dir),
            expected_sha=match.expected_bundle_sha, observed_sha=observed,
            kind="legacy_bundle",
        ))
    return QuarantineReceiptV1(
        scan_sha=scan.artifact_sha, quarantine_root=str(quarantine),
        copies=tuple(copies),
    )


def build_pinned_download_specs(
    project_root: Path | str,
) -> tuple[tuple[PinnedDownloadSpecV1, ...], tuple[str, ...]]:
    """Resolve only official, immutable-or-SHA-gated automatic source locators."""
    recovery = build_data_recovery_manifest(project_root)
    monash_revision = "7bf79ee8270e340b6c5848b7b56d8e1c35305fb6"
    sensor_commit = "82922c830800ca7aeaf53acc412a6d2cf7e56055"
    specs: list[PinnedDownloadSpecV1] = []
    manual: set[str] = set()
    for asset in recovery.assets:
        if asset.status == "PRESENT_VERIFIED":
            continue
        candidates = asset.candidate_paths
        source_id = asset.source_id
        url: str | None = None
        basename = Path(candidates[0]).name if candidates else asset.expected_sha
        if source_id == "monash_hf" and candidates:
            marker = "data/benchmark_v0/raw/monash_hf/"
            relative = next(path.split(marker, 1)[1] for path in candidates if marker in path)
            basename = Path(relative).name
            url = (
                "https://huggingface.co/datasets/monash_tsf/resolve/"
                f"{monash_revision}/{relative}?download=true"
            )
        elif source_id == "noaa_global_hourly" and candidates:
            if basename == "isd-history.csv":
                url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
            else:
                url = f"https://www.ncei.noaa.gov/data/global-hourly/access/2024/{basename}"
        elif source_id == "uci_electricity_load_diagrams":
            url = (
                "https://archive.ics.uci.edu/static/public/321/"
                "electricityloaddiagrams20112014.zip"
            )
        elif source_id == "metr_la":
            url = (
                "https://drive.usercontent.google.com/download?"
                "id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX&export=download&confirm=t"
            )
        elif source_id == "metr_la_spatial_pin":
            url = (
                "https://raw.githubusercontent.com/liyaguang/DCRNN/"
                f"{sensor_commit}/data/sensor_graph/graph_sensor_locations.csv"
            )
        else:
            manual.add(source_id)
        if url is not None:
            specs.append(PinnedDownloadSpecV1(
                source_id=source_id, url=url, basename=basename,
                expected_sha=asset.expected_sha,
                canonical_candidates=candidates,
            ))
    unique = {(row.expected_sha, row.url): row for row in specs}
    return (
        tuple(sorted(unique.values(), key=lambda row: (row.source_id, row.expected_sha))),
        tuple(sorted(manual)),
    )


def build_pinned_download_plan(project_root: Path | str) -> PinnedDownloadPlanV1:
    recovery = build_data_recovery_manifest(project_root)
    specs, manual = build_pinned_download_specs(project_root)
    return PinnedDownloadPlanV1(
        recovery_manifest_sha=recovery.artifact_sha,
        specs=specs, manual_only_sources=manual,
    )


def download_pinned_assets(
    project_root: Path | str,
    quarantine_root: Path | str,
    *,
    timeout_seconds: float = 60.0,
) -> PinnedDownloadReceiptV1:
    plan = build_pinned_download_plan(project_root)
    quarantine = Path(quarantine_root).resolve()
    results: list[PinnedDownloadResultV1] = []
    for spec in plan.specs:
        destination = quarantine / "downloads" / spec.expected_sha / spec.basename
        if destination.is_file() and file_sha256(destination) == spec.expected_sha:
            results.append(PinnedDownloadResultV1(
                source_id=spec.source_id, url=spec.url, expected_sha=spec.expected_sha,
                observed_sha=spec.expected_sha, size_bytes=destination.stat().st_size,
                status="EXACT_QUARANTINED", destination=str(destination),
            ))
            continue
        temporary = quarantine / ".partial" / f"{spec.expected_sha}.{spec.basename}"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            temporary.unlink()
        try:
            request = urllib.request.Request(
                spec.url, headers={"User-Agent": "TSharness-vNext-M0-Recovery/1"},
            )
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            observed = digest.hexdigest()
            if observed != spec.expected_sha:
                temporary.unlink(missing_ok=True)
                results.append(PinnedDownloadResultV1(
                    source_id=spec.source_id, url=spec.url,
                    expected_sha=spec.expected_sha, observed_sha=observed,
                    size_bytes=size, status="SHA_MISMATCH", destination=None,
                ))
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
            results.append(PinnedDownloadResultV1(
                source_id=spec.source_id, url=spec.url,
                expected_sha=spec.expected_sha, observed_sha=file_sha256(destination),
                size_bytes=size, status="EXACT_QUARANTINED", destination=str(destination),
            ))
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            results.append(PinnedDownloadResultV1(
                source_id=spec.source_id, url=spec.url,
                expected_sha=spec.expected_sha, observed_sha=None, size_bytes=None,
                status="DOWNLOAD_ERROR", destination=None,
                error=f"{type(exc).__name__}:{exc}",
            ))
    return PinnedDownloadReceiptV1(
        recovery_manifest_sha=plan.recovery_manifest_sha,
        results=tuple(results), manual_only_sources=plan.manual_only_sources,
    )


def promote_quarantined_legacy_bundle(
    project_root: Path | str,
    quarantine_root: Path | str,
) -> LegacyPromotionReceiptV1:
    """Promote the one frozen legacy bundle, retaining displaced bytes in quarantine."""
    project = Path(project_root).resolve()
    quarantine = Path(quarantine_root).resolve()
    recovery = build_data_recovery_manifest(project)
    expected = recovery.legacy_registry_binding_sha
    if expected is None:
        raise RuntimeError("frozen registry has no unique legacy bundle binding")
    source = quarantine / "legacy" / expected
    if legacy_bundle_sha(source) != expected:
        raise RuntimeError("quarantine does not contain the frozen legacy bundle")
    destination = project / "data" / "_artifacts"
    previous = legacy_bundle_sha(destination)
    displaced: Path | None = None
    if previous is not None and previous != expected:
        displaced = quarantine / "displaced_legacy" / previous
        _copy_exact(
            destination / LEGACY_META, displaced / LEGACY_META,
            file_sha256(destination / LEGACY_META),
        )
        _copy_exact(
            destination / LEGACY_VALUES, displaced / LEGACY_VALUES,
            file_sha256(destination / LEGACY_VALUES),
        )
        if legacy_bundle_sha(displaced) != previous:
            raise IOError("displaced legacy backup failed composite verification")
    _replace_exact(
        source / LEGACY_META, destination / LEGACY_META,
        file_sha256(source / LEGACY_META),
    )
    _replace_exact(
        source / LEGACY_VALUES, destination / LEGACY_VALUES,
        file_sha256(source / LEGACY_VALUES),
    )
    promoted = legacy_bundle_sha(destination)
    if promoted != expected:
        raise IOError("promoted legacy bundle failed composite verification")
    return LegacyPromotionReceiptV1(
        expected_bundle_sha=expected, previous_bundle_sha=previous,
        promoted_bundle_sha=promoted, source_directory=str(source),
        destination_directory=str(destination),
        displaced_backup_directory=str(displaced) if displaced is not None else None,
    )


def promote_verified_clean_base(
    project_root: Path | str,
    source_root: Path | str,
) -> CleanBasePromotionReceiptV1:
    """Copy a fully verified candidate to the canonical v0.2 derived-data path."""
    project = Path(project_root).resolve()
    source = Path(source_root).resolve()
    destination = project / "data" / "benchmark_v0_2" / "clean_base"
    source_manifest = verify_clean_base_at(project, source)
    if not source_manifest.ready or source_manifest.expected_record_count != 1919:
        raise RuntimeError("source clean-base is not a complete frozen-v0.2 match")
    source_sha = clean_base_content_sha(source_manifest)

    if destination.exists():
        destination_manifest = verify_clean_base_at(project, destination)
        if not destination_manifest.ready:
            raise FileExistsError("canonical v0.2 clean-base exists but is not verified")
        destination_sha = clean_base_content_sha(destination_manifest)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".clean_base.promote.{source_sha}.tmp"
        if staging.exists():
            raise FileExistsError(f"stale clean-base promotion staging exists: {staging}")
        shutil.copytree(source, staging, symlinks=False)
        staging_manifest = verify_clean_base_at(project, staging)
        if not staging_manifest.ready:
            raise IOError("copied clean-base failed frozen registry verification")
        if clean_base_content_sha(staging_manifest) != source_sha:
            raise IOError("copied clean-base content binding differs from source")
        os.replace(staging, destination)
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:  # pragma: no cover
            pass
        destination_manifest = verify_clean_base_at(project, destination)
        destination_sha = clean_base_content_sha(destination_manifest)
    if source_sha != destination_sha:
        raise IOError("canonical clean-base differs from verified recovery source")
    return CleanBasePromotionReceiptV1(
        source_root=str(source), destination_root=str(destination),
        registry_sha=source_manifest.registry_sha,
        source_content_sha=source_sha, destination_content_sha=destination_sha,
        verified_records=source_manifest.expected_record_count,
    )
