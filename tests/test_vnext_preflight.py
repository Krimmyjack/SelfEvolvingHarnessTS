from __future__ import annotations

from pathlib import Path

from SelfEvolvingHarnessTS.vnext.preflight import (
    CleanBaseIntegrityManifestV1,
    DataRecoveryManifestV1,
    EnvironmentProbeV1,
    EnvironmentLockV1,
    M0LayeredReproductionV1,
    PUBLISHED_V02_HEADLINES,
    ReproductionLayerV1,
    assess_m0_readiness,
    build_data_recovery_manifest,
    default_v02_preflight,
    decide_m0,
    probe_environment,
    run_preflight,
    verify_clean_base,
)


ROOT = Path(__file__).resolve().parents[1]


def test_integrity_check_cannot_claim_m0_and_detects_exact_drift(tmp_path):
    observed = dict(PUBLISHED_V02_HEADLINES)
    observed["raw"] += 2e-9
    report = run_preflight(
        tmp_path, required_paths=("split_manifest.json",), observed_headlines=observed,
    )
    assert not report.ok
    assert not report.m0_pass_eligible
    assert report.missing_paths == ("split_manifest.json",)
    assert any(item.startswith("raw:") for item in report.baseline_mismatches)


def test_repository_integrity_matches_after_exact_data_recovery():
    report = default_v02_preflight(ROOT)
    assert report.sha_mismatches == ()
    assert report.baseline_mismatches == ()
    assert report.missing_paths == ()
    assert report.ok
    assert report.report_kind == "artifact_integrity_only"


def test_current_checkout_stays_machine_readable_m0_blocked_before_reproduction():
    recovery = build_data_recovery_manifest(ROOT)
    clean = verify_clean_base(ROOT)
    environment = probe_environment()
    readiness = assess_m0_readiness(recovery, clean, environment)
    assert recovery.acquisition_coverage_missing_shas
    assert recovery.legacy_binding_matches
    assert len(clean.records) == 1919
    assert clean.ready
    assert readiness.status == "M0_BLOCKED"
    assert not readiness.task_g_authorized
    assert recovery.ready
    assert "data_recovery_not_ready" not in readiness.blockers
    assert "clean_base_not_ready" not in readiness.blockers
    assert readiness.blockers == ("environment_lock_mismatch",)


def test_environment_spec_and_real_uv_lock_are_both_bound():
    lock = EnvironmentLockV1.from_project(ROOT)
    assert lock.environment_spec_sha is not None
    assert lock.uv_lock_sha is not None
    probe = probe_environment(lock)
    assert "uv_lock:missing" not in probe.mismatches


def test_cli_m0_audit_cannot_ignore_an_open_protocol_erratum(tmp_path):
    from SelfEvolvingHarnessTS.vnext.cli import m0_audit

    # The repository's real readiness inputs are used; only artifact writes go to tmp.
    result = m0_audit(ROOT, tmp_path)
    assert result["status"] == "M0_BLOCKED"
    assert "open_protocol_erratum" in result["blockers"]
    assert not result["task_g_authorized"]


def test_final_m0_verdict_requires_and_accepts_complete_shadow_reproduction():
    recovery = DataRecoveryManifestV1(
        benchmark_version="benchmark-v0.2", acquisition_manifest_sha="1" * 64,
        registry_sha="2" * 64, benchmark_manifest_sha="3" * 64, assets=(),
        acquisition_coverage_missing_shas=(), legacy_registry_binding_sha="4" * 64,
        local_legacy_bundle_sha="4" * 64, legacy_binding_matches=True,
    )
    clean = CleanBaseIntegrityManifestV1(
        registry_sha="2" * 64, clean_base_root="shadow/clean_base",
        records=(), expected_record_count=0,
    )
    environment = EnvironmentProbeV1(
        python_version="3.10.0", implementation="CPython", platform_system="Darwin",
        platform_release="test", platform_machine="arm64", packages=(), environment=(),
        lock_sha="5" * 64, lock_matches=True, mismatches=(),
    )
    reproduction = M0LayeredReproductionV1(
        data_recovery_sha=recovery.artifact_sha, clean_base_sha=clean.artifact_sha,
        environment_probe_sha=environment.artifact_sha, shadow_root="shadow",
        observed_source="shadow_raw_to_result_reproduction",
        layers=tuple(ReproductionLayerV1(f"L{i}", True, "6" * 64, "6" * 64) for i in range(10)),
        observed_headlines=PUBLISHED_V02_HEADLINES,
        per_uid_loss_digest="7" * 64, program_provenance_digest="8" * 64,
    )
    verdict = decide_m0(recovery, clean, environment, reproduction)
    assert verdict.verdict == "M0_PASS"
    assert verdict.task_g_authorized
