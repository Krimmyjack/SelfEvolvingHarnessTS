"""Contract tests for the benchmark-v0 outer split manifest."""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from SelfEvolvingHarnessTS.benchmark import (
    BENCHMARK_VERSION,
    DESIGN_COMMIT,
    EXTERNAL_ADDENDUM_SHA256,
    HEADLINE_HORIZON,
    HEADLINE_LOOKBACK,
    HEADLINE_MIN_LENGTH,
)
from SelfEvolvingHarnessTS.benchmark.split import (
    SplitCandidate,
    SplitManifest,
    SplitManifestError,
    SplitRole,
    build_split_manifest,
    role_from_unit_interval,
    validate_split_manifest,
)


def _candidate(
    uid: str,
    *,
    group: str | None = None,
    exposure: str = "certified_virgin",
    length: int | None = None,
) -> SplitCandidate:
    return SplitCandidate(uid, "dataset", "regime", group, exposure, length)


def test_dev_query_is_independent_repeatable_role():
    roles = role_from_unit_interval
    assert roles(0.0) is SplitRole.SUPPORT_A
    assert roles(0.24) is SplitRole.SUPPORT_A
    assert roles(0.25) is SplitRole.SUPPORT_B
    assert roles(0.45) is SplitRole.DEV_QUERY
    assert roles(0.65) is SplitRole.FINAL_QUERY
    assert roles(0.999999) is SplitRole.FINAL_QUERY
    for bad in (-0.001, 1.0, float("nan")):
        with pytest.raises(ValueError, match=r"\[0,1\)"):
            roles(bad)


def test_overlap_group_is_atomic_and_forced_roles_win():
    rows = [
        SplitCandidate("a", "d", "r", "g", "certified_virgin"),
        SplitCandidate("b", "d", "r", "g", "certified_virgin"),
        SplitCandidate("legacy", "d", "r", "legacy-g", "confirmed_exposed"),
        SplitCandidate("u-a", "d", "r", "u-g", "certified_virgin"),
        SplitCandidate("u-b", "d", "r", "u-g", "certified_virgin"),
    ]
    manifest = build_split_manifest(
        rows, "benchmark-v0", "split-salt-v0", {"u-a"}
    )
    assert manifest.assignment("a").role == manifest.assignment("b").role
    assert manifest.assignment("legacy").role is SplitRole.SUPPORT_A
    assert manifest.assignment("u-a").role is SplitRole.U
    assert manifest.assignment("u-b").role is SplitRole.U


def test_support_a_and_u_force_conflict_is_rejected():
    rows = [
        _candidate("legacy", group="mixed", exposure="confirmed_exposed"),
        _candidate("selected-u", group="mixed"),
    ]
    with pytest.raises(SplitManifestError, match="Support-A.*U"):
        build_split_manifest(rows, "benchmark-v0", "salt", {"selected-u"})


def test_dev_query_policy_cannot_alias_support_b():
    policy = SplitManifest.role_policies()[SplitRole.DEV_QUERY]
    assert policy == {
        "repeatable": True,
        "utility_visible": True,
        "may_select_best_fixed": False,
        "may_train_oracle_transfer": False,
        "may_confirm_method": False,
        "final_eligible": False,
    }
    assert policy != SplitManifest.role_policies()[SplitRole.SUPPORT_B]


def test_manifest_is_reorder_deterministic_and_validates_atomicity():
    rows = [
        _candidate("c", group="g-2"),
        _candidate("a", group="g-1"),
        _candidate("b", group="g-1"),
    ]
    forward = build_split_manifest(rows, BENCHMARK_VERSION, "salt", set())
    reverse = build_split_manifest(list(reversed(rows)), BENCHMARK_VERSION, "salt", set())
    assert forward.manifest_sha == reverse.manifest_sha
    assert forward.assignments == reverse.assignments
    validate_split_manifest(forward)

    changed = dataclasses.replace(
        forward.assignments[0], role=SplitRole.U
    )
    tampered = dataclasses.replace(
        forward, assignments=(changed, *forward.assignments[1:])
    )
    with pytest.raises(SplitManifestError):
        validate_split_manifest(tampered)


def test_manifest_records_chronological_boundaries_and_design_provenance():
    manifest = build_split_manifest(
        [_candidate("long", length=HEADLINE_MIN_LENGTH)],
        BENCHMARK_VERSION,
        "salt",
        set(),
    )
    bounds = manifest.assignment("long").chronological_boundaries
    assert bounds == {
        "train": [0, HEADLINE_MIN_LENGTH - 2 * HEADLINE_HORIZON],
        "validation": [
            HEADLINE_MIN_LENGTH - 2 * HEADLINE_HORIZON,
            HEADLINE_MIN_LENGTH - HEADLINE_HORIZON,
        ],
        "test": [HEADLINE_MIN_LENGTH - HEADLINE_HORIZON, HEADLINE_MIN_LENGTH],
    }
    assert manifest.inner_split == {
        "indexing": "zero_based_half_open",
        "lookback": HEADLINE_LOOKBACK,
        "horizon": HEADLINE_HORIZON,
        "validation_size": HEADLINE_HORIZON,
        "test_size": HEADLINE_HORIZON,
        "minimum_length": HEADLINE_MIN_LENGTH,
    }
    assert manifest.provenance["external_addendum_sha256"] == EXTERNAL_ADDENDUM_SHA256
    assert manifest.provenance["design_commit"] == DESIGN_COMMIT


def test_invalid_candidates_and_u_selection_are_rejected():
    with pytest.raises(SplitManifestError, match="duplicate series_uid"):
        build_split_manifest(
            [_candidate("same"), _candidate("same")], BENCHMARK_VERSION, "salt", set()
        )
    with pytest.raises(SplitManifestError, match="unknown U-selected"):
        build_split_manifest([_candidate("known")], BENCHMARK_VERSION, "salt", {"missing"})
    with pytest.raises(SplitManifestError, match="minimum"):
        build_split_manifest(
            [_candidate("short", length=HEADLINE_MIN_LENGTH - 1)],
            BENCHMARK_VERSION,
            "salt",
            set(),
        )


def test_repository_addendum_is_an_exact_byte_mirror():
    mirror = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmark"
        / "Benchmark_v0_Forecast_Design_v3_Addendum_2026-07-13.md"
    )
    assert hashlib.sha256(mirror.read_bytes()).hexdigest() == EXTERNAL_ADDENDUM_SHA256
