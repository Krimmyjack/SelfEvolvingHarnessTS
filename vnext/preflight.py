"""M0 recovery, environment, and raw-to-result reproduction contracts.

Artifact integrity and scientific reproduction are deliberately separate.  Reading a
published report can prove that frozen bytes are intact; it can never produce M0_PASS.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..benchmark.registry import SeriesRecord
from ._canonical import file_sha256, require_sha, sha256


PUBLISHED_V02_HEADLINES = {
    "raw": 11.481494078943095,
    "best_fixed_stl": 10.978415119273457,
    "h_ref": 11.558341964549324,
    "transfer_retrained_ceiling": 10.78812516546172,
    "ex_covid_raw": 1.0300309663410663,
    "ex_covid_stl": 0.9400922511500845,
    "ex_covid_ceiling": 0.898147853060101,
}
PUBLISHED_V02_TOLERANCES = {name: 1e-9 for name in PUBLISHED_V02_HEADLINES}

RECOVERY_STATES = frozenset({
    "PRESENT_VERIFIED", "PRESENT_SHA_MISMATCH", "MISSING_RECOVERABLE",
    "MISSING_NO_SOURCE", "REGENERATED_MATCH", "REGENERATED_MISMATCH",
})


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@dataclass(frozen=True)
class PreflightReport:
    """Frozen artifact integrity report; never an M0 scientific verdict."""

    ok: bool
    missing_paths: tuple[str, ...]
    sha_mismatches: tuple[str, ...]
    dependency_fingerprint: Mapping[str, str]
    baseline_mismatches: tuple[str, ...]
    m0_pass_eligible: bool = False
    report_kind: str = "artifact_integrity_only"

    @property
    def sha256(self) -> str:
        return sha256(self)


def verify_headlines(observed: Mapping[str, float]) -> tuple[str, ...]:
    failures = []
    for name, expected in PUBLISHED_V02_HEADLINES.items():
        if name not in observed:
            failures.append(f"missing:{name}")
        elif abs(float(observed[name]) - expected) > PUBLISHED_V02_TOLERANCES[name]:
            failures.append(f"{name}:{observed[name]}!={expected}")
    return tuple(failures)


def _package_fingerprint() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in (
        "numpy", "scipy", "statsmodels", "PyWavelets", "scikit-learn",
        "pandas", "pyarrow", "h5py", "tables", "joblib", "requests",
        "torch", "chronos-forecasting",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "missing"
    return result


def run_preflight(
    root: Path | str,
    *,
    required_paths: Sequence[str],
    expected_shas: Mapping[str, str] | None = None,
    observed_headlines: Mapping[str, float] | None = None,
) -> PreflightReport:
    root = Path(root)
    missing = tuple(sorted(path for path in required_paths if not (root / path).exists()))
    mismatches = []
    for relative, expected in sorted((expected_shas or {}).items()):
        path = root / relative
        if path.exists() and file_sha256(path) != expected:
            mismatches.append(relative)
    deps = {"python": platform.python_version(), **_package_fingerprint()}
    baselines = () if observed_headlines is None else verify_headlines(observed_headlines)
    return PreflightReport(
        ok=not missing and not mismatches and not baselines,
        missing_paths=missing,
        sha_mismatches=tuple(mismatches),
        dependency_fingerprint=deps,
        baseline_mismatches=baselines,
    )


def default_v02_preflight(project_root: Path | str) -> PreflightReport:
    """Check frozen bindings and data presence without pretending to reproduce results."""
    root = Path(project_root)
    result_root = root / "results" / "Benchmark_v0_2"
    manifest = _json(result_root / "benchmark_manifest_v0.yaml")
    expected = {
        "data/benchmark_v0/acquisition_manifest.json": manifest["acquisition_manifest_sha256"],
        "results/Benchmark_v0_2/split_manifest.json": manifest["split_manifest_sha256"],
        "results/Benchmark_v0_2/support_a_subsplit.json": manifest["support_a_subsplit_sha256"],
        "results/Benchmark_v0_2/corruption_grid.json": manifest["corruption_grid_sha256"],
        "results/Benchmark_v0_2/dataset_manifest.json": manifest["dataset_manifest_sha256"],
        "results/Benchmark_v0_2/program_pool.json": manifest["program_pool_sha256"],
        "results/Benchmark_v0_2/series_registry.jsonl": manifest["registry_sha256"],
        "results/Benchmark_v0_2/metr_la_spatial_blocks.json": manifest[
            "metr_la_spatial_blocks_sha256"
        ],
    }
    required = tuple(expected) + (
        "data/benchmark_v0/raw", "data/benchmark_v0_2/clean_base",
    )
    return run_preflight(root, required_paths=required, expected_shas=expected)


@dataclass(frozen=True)
class RecoveryAssetV1:
    source_id: str
    expected_sha: str
    candidate_paths: tuple[str, ...]
    actual_path: str | None
    actual_sha: str | None
    size_bytes: int | None
    status: str
    authority_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha(self.expected_sha, "expected_sha")
        if self.actual_sha is not None:
            require_sha(self.actual_sha, "actual_sha")
        if self.status not in RECOVERY_STATES:
            raise ValueError("unknown recovery status")


@dataclass(frozen=True)
class DataRecoveryManifestV1:
    benchmark_version: str
    acquisition_manifest_sha: str
    registry_sha: str
    benchmark_manifest_sha: str
    assets: tuple[RecoveryAssetV1, ...]
    acquisition_coverage_missing_shas: tuple[str, ...]
    legacy_registry_binding_sha: str | None
    local_legacy_bundle_sha: str | None
    legacy_binding_matches: bool
    schema_version: str = "vnext-data-recovery/1"

    def __post_init__(self) -> None:
        for name in (
            "acquisition_manifest_sha", "registry_sha", "benchmark_manifest_sha",
        ):
            require_sha(getattr(self, name), name)
        for name in ("legacy_registry_binding_sha", "local_legacy_bundle_sha"):
            value = getattr(self, name)
            if value is not None:
                require_sha(value, name)

    @property
    def ready(self) -> bool:
        return (
            self.legacy_binding_matches
            and all(asset.status in {"PRESENT_VERIFIED", "REGENERATED_MATCH"} for asset in self.assets)
        )

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


def _normalise_recorded_path(project_root: Path, value: str) -> str:
    parts = Path(value).parts
    if parts and parts[0] == project_root.name:
        return str(Path(*parts[1:]))
    return str(Path(value))


def _legacy_bundle_sha(project_root: Path) -> str | None:
    """Reproduce the legacy inventory binding used by benchmark.workspace."""
    meta = project_root / "data" / "_artifacts" / "monash_clean.meta.jsonl"
    values = project_root / "data" / "_artifacts" / "monash_clean.npz"
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


def build_data_recovery_manifest(project_root: Path | str) -> DataRecoveryManifestV1:
    root = Path(project_root)
    acquisition_path = root / "data" / "benchmark_v0" / "acquisition_manifest.json"
    result = root / "results" / "Benchmark_v0_2"
    registry_path = result / "series_registry.jsonl"
    benchmark_path = result / "benchmark_manifest_v0.yaml"
    spatial_path = result / "metr_la_spatial_blocks.json"
    acquisition = _json(acquisition_path)
    registry = list(_jsonl(registry_path))
    acquisition_shas = {
        expected_sha
        for item in acquisition["results"]
        for expected_sha in item.get("asset_sha256", ())
    }

    paths_by_sha: dict[str, set[str]] = {}
    source_by_sha: dict[str, set[str]] = {}
    for item in acquisition["results"]:
        source_id = str(item["source_id"])
        shas = list(item.get("asset_sha256", ()))
        paths = list(item.get("asset_paths", ()))
        for index, expected_sha in enumerate(shas):
            source_by_sha.setdefault(expected_sha, set()).add(source_id)
            if index < len(paths):
                paths_by_sha.setdefault(expected_sha, set()).add(
                    _normalise_recorded_path(root, paths[index])
                )
    registry_sources: dict[str, set[str]] = {}
    registry_entities: dict[str, set[str]] = {}
    for row in registry:
        expected_sha = str(row["source_asset_sha256"])
        registry_sources.setdefault(expected_sha, set()).add(str(row["source_id"]))
        registry_entities.setdefault(expected_sha, set()).add(str(row["entity_id"]))

    # The original acquisition receipt only listed the NOAA files that happened to be
    # present when it was written.  The frozen registry remains authoritative for all
    # selected stations and contains the station id needed to reconstruct the canonical
    # raw path.  This is path recovery, not dataset resampling.
    for expected_sha, source_ids in registry_sources.items():
        if source_ids == {"noaa_global_hourly"}:
            for entity_id in registry_entities[expected_sha]:
                paths_by_sha.setdefault(expected_sha, set()).add(
                    f"data/benchmark_v0/raw/noaa_global_hourly/2024/{entity_id}.csv"
                )
                source_by_sha.setdefault(expected_sha, set()).add("frozen_registry_station")

    spatial = _json(spatial_path)
    spatial_source_sha = str(spatial["sensor_locations_sha256"])
    paths_by_sha.setdefault(spatial_source_sha, set()).add(
        "data/benchmark_v0/raw/metr_la/graph_sensor_locations.csv"
    )
    source_by_sha.setdefault(spatial_source_sha, set()).add("metr_la_spatial_pin")

    legacy_bindings = {
        row["source_asset_sha256"] for row in registry
        if str(row["source_id"]).startswith("legacy_")
    }
    legacy_registry = next(iter(legacy_bindings)) if len(legacy_bindings) == 1 else None
    local_legacy = _legacy_bundle_sha(root)

    expected_shas = sorted(set(paths_by_sha) | set(registry_sources))
    assets: list[RecoveryAssetV1] = []
    for expected_sha in expected_shas:
        candidates = tuple(sorted(paths_by_sha.get(expected_sha, ())))
        existing = [relative for relative in candidates if (root / relative).is_file()]
        actual_path = existing[0] if existing else None
        actual_sha = file_sha256(root / actual_path) if actual_path else None
        size = (root / actual_path).stat().st_size if actual_path else None
        is_legacy_bundle = expected_sha in legacy_bindings
        if is_legacy_bundle and local_legacy is not None:
            actual_path = "data/_artifacts/{monash_clean.meta.jsonl,monash_clean.npz}"
            actual_sha = local_legacy
            meta = root / "data" / "_artifacts" / "monash_clean.meta.jsonl"
            values = root / "data" / "_artifacts" / "monash_clean.npz"
            size = meta.stat().st_size + values.stat().st_size
            status = (
                "PRESENT_VERIFIED" if actual_sha == expected_sha
                else "PRESENT_SHA_MISMATCH"
            )
        elif actual_path:
            status = "PRESENT_VERIFIED" if actual_sha == expected_sha else "PRESENT_SHA_MISMATCH"
        else:
            status = "MISSING_RECOVERABLE" if candidates else "MISSING_NO_SOURCE"
        assets.append(RecoveryAssetV1(
            source_id="|".join(sorted(registry_sources.get(expected_sha) or source_by_sha.get(expected_sha) or {"unknown"})),
            expected_sha=expected_sha,
            candidate_paths=candidates,
            actual_path=actual_path,
            actual_sha=actual_sha,
            size_bytes=size,
            status=status,
            authority_sources=tuple(sorted(
                ({"frozen_registry"} if expected_sha in registry_sources else set())
                | ({"acquisition_manifest"} if expected_sha in acquisition_shas else set())
                | ({"metr_la_spatial_pin"} if expected_sha == spatial_source_sha else set())
            )),
        ))

    coverage_gap = tuple(sorted(set(registry_sources) - acquisition_shas))
    return DataRecoveryManifestV1(
        benchmark_version="benchmark-v0.2",
        acquisition_manifest_sha=file_sha256(acquisition_path),
        registry_sha=file_sha256(registry_path),
        benchmark_manifest_sha=file_sha256(benchmark_path),
        assets=tuple(assets),
        acquisition_coverage_missing_shas=coverage_gap,
        legacy_registry_binding_sha=legacy_registry,
        local_legacy_bundle_sha=local_legacy,
        legacy_binding_matches=(legacy_registry is not None and local_legacy == legacy_registry),
    )


@dataclass(frozen=True)
class CleanBaseRecordCheckV1:
    series_uid: str
    slot_key: str
    status: str
    failures: tuple[str, ...]


@dataclass(frozen=True)
class CleanBaseIntegrityManifestV1:
    registry_sha: str
    clean_base_root: str
    records: tuple[CleanBaseRecordCheckV1, ...]
    expected_record_count: int
    schema_version: str = "vnext-clean-base-integrity/1"

    def __post_init__(self) -> None:
        require_sha(self.registry_sha, "registry_sha")
        if len(self.records) != self.expected_record_count:
            raise ValueError("clean-base manifest does not cover the complete registry")

    @property
    def ready(self) -> bool:
        return all(row.status == "PRESENT_VERIFIED" for row in self.records)

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


def verify_clean_base_at(
    project_root: Path | str,
    clean_base_root: Path | str,
) -> CleanBaseIntegrityManifestV1:
    """Verify an arbitrary clean-base candidate against the frozen v0.2 registry."""
    root = Path(project_root)
    registry_path = root / "results" / "Benchmark_v0_2" / "series_registry.jsonl"
    clean_root = Path(clean_base_root)
    if not clean_root.is_absolute():
        clean_root = root / clean_root
    records: list[CleanBaseRecordCheckV1] = []
    for payload in _jsonl(registry_path):
        record = SeriesRecord.from_dict(payload)
        slot_key = hashlib.sha256(json.dumps(
            [record.source_id, record.dataset_id, record.entity_id],
            ensure_ascii=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        slot = clean_root / slot_key
        failures: list[str] = []
        required = [slot / "values.npy", slot / "natural_missing_mask.npy", slot / "record.json"]
        if record.timestamps_sha is not None:
            required.append(slot / "timestamps.npy")
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            failures.extend(f"missing:{name}" for name in missing)
        else:
            try:
                values = np.load(slot / "values.npy", allow_pickle=False)
                mask = np.load(slot / "natural_missing_mask.npy", allow_pickle=False)
                timestamps = (
                    np.load(slot / "timestamps.npy", allow_pickle=False)
                    if record.timestamps_sha is not None else None
                )
                record.verify_values(values, timestamps=timestamps)
                if not np.array_equal(mask.astype(bool), np.isnan(values)):
                    failures.append("mask_values_disagree")
                local_record = _json(slot / "record.json")
                for field_name in (
                    "source_id", "dataset_id", "entity_id", "source_asset_sha256",
                    "content_sha", "natural_missing_mask_sha", "timestamps_sha", "length",
                ):
                    if local_record.get(field_name) != payload.get(field_name):
                        failures.append(f"record_field:{field_name}")
            except Exception as exc:
                failures.append(f"verification:{type(exc).__name__}:{exc}")
        records.append(CleanBaseRecordCheckV1(
            series_uid=record.series_uid,
            slot_key=slot_key,
            status="PRESENT_VERIFIED" if not failures else "MISSING_OR_MISMATCH",
            failures=tuple(failures),
        ))
    return CleanBaseIntegrityManifestV1(
        registry_sha=file_sha256(registry_path),
        clean_base_root=str(clean_root),
        records=tuple(sorted(records, key=lambda row: row.series_uid)),
        expected_record_count=len(records),
    )


def verify_clean_base(project_root: Path | str) -> CleanBaseIntegrityManifestV1:
    return verify_clean_base_at(
        project_root, Path("data") / "benchmark_v0_2" / "clean_base",
    )


@dataclass(frozen=True)
class EnvironmentLockV1:
    python_version: str = "3.10.19"
    required_packages: tuple[tuple[str, str], ...] = (
        ("numpy", "2.2.6"), ("scipy", "1.15.2"),
        ("statsmodels", "0.14.6"), ("PyWavelets", "1.8.0"),
        ("scikit-learn", "1.7.2"),
        ("pandas", "2.3.3"), ("pyarrow", "19.0.1"),
        ("h5py", "3.13.0"), ("tables", "3.10.1"),
        ("joblib", "1.5.3"), ("requests", "2.32.5"),
        ("torch", "2.12.0"), ("chronos-forecasting", "2.3.0"),
    )
    platform_system: str = "Darwin"
    platform_machine: str = "arm64"
    omp_threads: int = 1
    mkl_threads: int = 1
    python_hash_seed: str = "20260713"
    timezone: str = "UTC"
    locale: str = "C"
    torch_cpu_only: bool = True
    torch_deterministic: bool = True
    uv_version: str = "0.11.28"
    environment_spec_sha: str | None = None
    uv_lock_sha: str | None = None
    schema_version: str = "vnext-environment-lock/1"

    def __post_init__(self) -> None:
        for name in ("environment_spec_sha", "uv_lock_sha"):
            value = getattr(self, name)
            if value is not None:
                require_sha(value, name)

    @classmethod
    def from_project(cls, project_root: Path | str) -> "EnvironmentLockV1":
        directory = Path(project_root) / "vnext" / "environment"
        spec = directory / "pyproject.toml"
        uv_lock = directory / "uv.lock"
        return cls(
            environment_spec_sha=file_sha256(spec) if spec.is_file() else None,
            uv_lock_sha=file_sha256(uv_lock) if uv_lock.is_file() else None,
        )

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class EnvironmentProbeV1:
    python_version: str
    implementation: str
    platform_system: str
    platform_release: str
    platform_machine: str
    packages: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str | None], ...]
    lock_sha: str
    lock_matches: bool
    mismatches: tuple[str, ...]
    schema_version: str = "vnext-environment-probe/1"

    def __post_init__(self) -> None:
        require_sha(self.lock_sha, "lock_sha")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


def probe_environment(lock: EnvironmentLockV1 | None = None) -> EnvironmentProbeV1:
    lock = lock or EnvironmentLockV1()
    packages = _package_fingerprint()
    mismatches: list[str] = []
    if lock.environment_spec_sha is None:
        mismatches.append("environment_spec:missing")
    if lock.uv_lock_sha is None:
        mismatches.append("uv_lock:missing")
    if platform.python_version() != lock.python_version:
        mismatches.append(f"python:{platform.python_version()}!={lock.python_version}")
    if platform.system() != lock.platform_system:
        mismatches.append(f"system:{platform.system()}!={lock.platform_system}")
    if platform.machine() != lock.platform_machine:
        mismatches.append(f"machine:{platform.machine()}!={lock.platform_machine}")
    for name, expected in lock.required_packages:
        actual = packages.get(name, "missing")
        if actual != expected:
            mismatches.append(f"{name}:{actual}!={expected}")
    env = tuple((name, os.environ.get(name)) for name in (
        "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TZ", "LC_ALL", "LANG",
    ))
    expected_env = {
        "PYTHONHASHSEED": lock.python_hash_seed,
        "OMP_NUM_THREADS": str(lock.omp_threads),
        "MKL_NUM_THREADS": str(lock.mkl_threads),
        "TZ": lock.timezone,
        "LC_ALL": lock.locale,
        "LANG": lock.locale,
    }
    for name, expected in expected_env.items():
        actual = dict(env).get(name)
        if actual != expected:
            mismatches.append(f"env:{name}:{actual}!={expected}")
    return EnvironmentProbeV1(
        python_version=platform.python_version(),
        implementation=platform.python_implementation(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        packages=tuple(sorted(packages.items())),
        environment=env,
        lock_sha=lock.artifact_sha,
        lock_matches=not mismatches,
        mismatches=tuple(mismatches),
    )


@dataclass(frozen=True)
class ReproductionLayerV1:
    layer: str
    passed: bool
    expected_sha: str | None
    observed_sha: str | None
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.layer not in {f"L{i}" for i in range(10)}:
            raise ValueError("unknown reproduction layer")
        for name in ("expected_sha", "observed_sha"):
            value = getattr(self, name)
            if value is not None:
                require_sha(value, name)


@dataclass(frozen=True)
class M0LayeredReproductionV1:
    data_recovery_sha: str
    clean_base_sha: str
    environment_probe_sha: str
    shadow_root: str
    observed_source: str
    layers: tuple[ReproductionLayerV1, ...]
    observed_headlines: Mapping[str, float]
    per_uid_loss_digest: str
    program_provenance_digest: str
    schema_version: str = "vnext-m0-layered-reproduction/1"

    def __post_init__(self) -> None:
        for name in (
            "data_recovery_sha", "clean_base_sha", "environment_probe_sha",
            "per_uid_loss_digest", "program_provenance_digest",
        ):
            require_sha(getattr(self, name), name)
        if self.observed_source != "shadow_raw_to_result_reproduction":
            raise ValueError("published reports cannot serve as M0 observed results")
        if tuple(row.layer for row in self.layers) != tuple(f"L{i}" for i in range(10)):
            raise ValueError("M0 reproduction must report L0 through L9 in order")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class M0ReproductionVerdictV1:
    reproduction_sha: str | None
    data_recovery_sha: str
    clean_base_sha: str
    environment_probe_sha: str
    observed_metrics: Mapping[str, float]
    absolute_errors: Mapping[str, float]
    verdict: str
    failure_reasons: tuple[str, ...]
    task_g_authorized: bool
    tolerance: float = 1e-9
    schema_version: str = "vnext-m0-verdict/1"

    def __post_init__(self) -> None:
        for name in ("data_recovery_sha", "clean_base_sha", "environment_probe_sha"):
            require_sha(getattr(self, name), name)
        if self.reproduction_sha is not None:
            require_sha(self.reproduction_sha, "reproduction_sha")
        allowed = {"M0_PASS", "M0_FAIL_PROTOCOL_ERRATUM_REQUIRED"}
        if self.verdict not in allowed:
            raise ValueError("M0 has no partial-pass state")
        if self.task_g_authorized != (self.verdict == "M0_PASS"):
            raise ValueError("Task G authorization must equal the M0 verdict")
        if self.verdict == "M0_PASS" and self.failure_reasons:
            raise ValueError("passing M0 cannot contain failure reasons")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class M0ReadinessStatusV1:
    data_recovery_sha: str
    clean_base_sha: str
    environment_probe_sha: str
    status: str
    blockers: tuple[str, ...]
    task_g_authorized: bool = False
    schema_version: str = "vnext-m0-readiness/1"

    def __post_init__(self) -> None:
        for name in ("data_recovery_sha", "clean_base_sha", "environment_probe_sha"):
            require_sha(getattr(self, name), name)
        if self.status not in {"M0_BLOCKED", "M0_READY_FOR_REPRODUCTION"}:
            raise ValueError("unknown M0 readiness state")
        if self.task_g_authorized:
            raise ValueError("readiness status can never authorize Task G")
        if self.status == "M0_BLOCKED" and not self.blockers:
            raise ValueError("blocked M0 readiness requires blockers")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


def assess_m0_readiness(
    recovery: DataRecoveryManifestV1,
    clean: CleanBaseIntegrityManifestV1,
    environment: EnvironmentProbeV1,
) -> M0ReadinessStatusV1:
    blockers: list[str] = []
    if not recovery.ready:
        blockers.append("data_recovery_not_ready")
    if not clean.ready:
        blockers.append("clean_base_not_ready")
    if not environment.lock_matches:
        blockers.append("environment_lock_mismatch")
    return M0ReadinessStatusV1(
        data_recovery_sha=recovery.artifact_sha,
        clean_base_sha=clean.artifact_sha,
        environment_probe_sha=environment.artifact_sha,
        status="M0_BLOCKED" if blockers else "M0_READY_FOR_REPRODUCTION",
        blockers=tuple(blockers),
    )


def decide_m0(
    recovery: DataRecoveryManifestV1,
    clean: CleanBaseIntegrityManifestV1,
    environment: EnvironmentProbeV1,
    reproduction: M0LayeredReproductionV1,
) -> M0ReproductionVerdictV1:
    failures: list[str] = []
    if not recovery.ready:
        failures.append("data_recovery_not_ready")
    if not clean.ready:
        failures.append("clean_base_not_ready")
    if not environment.lock_matches:
        failures.append("environment_lock_mismatch")
    if not isinstance(reproduction, M0LayeredReproductionV1):
        raise TypeError("M0 final verdict requires a shadow raw-to-result reproduction")
    observed: Mapping[str, float] = reproduction.observed_headlines
    errors: dict[str, float] = {}
    failures.extend(
        f"reproduction_layer_failed:{row.layer}" for row in reproduction.layers if not row.passed
    )
    headline_failures = verify_headlines(observed)
    failures.extend(headline_failures)
    for name, expected in PUBLISHED_V02_HEADLINES.items():
        if name in observed:
            errors[name] = abs(float(observed[name]) - expected)
    passed = not failures
    return M0ReproductionVerdictV1(
        reproduction_sha=reproduction.artifact_sha,
        data_recovery_sha=recovery.artifact_sha,
        clean_base_sha=clean.artifact_sha,
        environment_probe_sha=environment.artifact_sha,
        observed_metrics=dict(observed),
        absolute_errors=errors,
        verdict="M0_PASS" if passed else "M0_FAIL_PROTOCOL_ERRATUM_REQUIRED",
        failure_reasons=tuple(sorted(set(failures))),
        task_g_authorized=passed,
    )
