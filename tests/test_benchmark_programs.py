"""Guards for the three faces benchmark-v0.2 freezes.

Each test here corresponds to a way this project has actually been burned before:
an operator that silently became a different operator, a training pool that leaked a
private label, and a validation set that had already fed the thing it was validating.
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark import BENCHMARK_VERSION
from SelfEvolvingHarnessTS.benchmark.dev_eval import _evaluate_role, _fit_and_score
from SelfEvolvingHarnessTS.benchmark.programs import (
    CAPABILITY_GAPS,
    PROGRAM_IDS,
    PoolProgram,
    ProgramPoolError,
    apply_program,
    mechanism_of,
    pool_manifest,
    run_pool_with_provenance,
)
from SelfEvolvingHarnessTS.benchmark.method_api import MethodSeriesView
from SelfEvolvingHarnessTS.benchmark.split import (
    SplitManifestError,
    SplitRole,
    build_support_a_subsplit,
)
from SelfEvolvingHarnessTS.benchmark.trainers import NormalizationState


def test_version_is_v0_2_and_older_arenas_stay_readable():
    from SelfEvolvingHarnessTS.benchmark import KNOWN_BENCHMARK_VERSIONS

    assert BENCHMARK_VERSION == "benchmark-v0.2"
    # Sealed Final-Query sets of the older arenas must remain auditable.
    assert "benchmark-v0" in KNOWN_BENCHMARK_VERSIONS
    assert "benchmark-v0.1" in KNOWN_BENCHMARK_VERSIONS


def test_pool_covers_every_defect_mechanism_it_claims_and_declares_the_rest():
    manifest = pool_manifest()
    by_mechanism = manifest["programs_by_mechanism"]
    # The v0.1 pool was all fills, which is precisely why five of the nine corruption cells
    # returned identical losses for every program.
    assert set(by_mechanism) == {
        "none",
        "missing",
        "outlier_spike",
        "additive_noise",
        "reference",
    }
    assert len(by_mechanism["outlier_spike"]) >= 2
    assert len(by_mechanism["additive_noise"]) >= 2

    gaps = {gap["defect_mechanism"] for gap in manifest["capability_gaps"]}
    assert gaps == {"timestamp_irregularity", "structural_break"}
    # A gap must name the corruption scenario it fails to answer, or it is not auditable.
    for gap in CAPABILITY_GAPS:
        assert gap["corruption_scenario"]
        assert gap["why_uncovered"]


def test_pool_pins_the_code_that_implements_it():
    manifest = pool_manifest()
    digests = manifest["code_sha256"]
    # Operator behaviour lives in these files. If any of them moves, the pool's identity --
    # and therefore every number attributed to it -- must be considered stale.
    for path in (
        "benchmark/programs.py",
        "operators/registry.py",
        "operators/s1_denoise.py",
        "operators/s1_outlier.py",
    ):
        assert len(digests[path]) == 64
    assert manifest["frozen_before_any_v0_2_result_was_read"] is True
    assert manifest["dependency_fingerprint"]["scipy"] != "MISSING"


def test_no_operator_masquerades_as_another_on_a_realistic_series():
    # denoise_stl and denoise_wavelet both fall back to savgol on their failure branches.
    # A pool member that quietly becomes a different operator forges exactly the
    # conditioning signal this benchmark exists to measure, so the ledger must come back
    # clean on the shapes the roster actually contains.
    rng = np.random.default_rng(0)
    t = np.arange(512, dtype=float)
    values = 10.0 + np.sin(2 * np.pi * t / 24.0) + 0.1 * rng.standard_normal(512)
    values[100:112] = np.nan
    values[300] = 90.0

    outputs, summary = run_pool_with_provenance(values, period=24)
    masquerades = {
        (requested, effective)
        for requested, by_effective in summary.items()
        for effective in by_effective
        if effective != requested
    }
    assert masquerades == set()
    for program_id, prepared in outputs.items():
        assert len(prepared) == len(values), program_id


def test_every_pool_program_keeps_length_units_and_finiteness():
    values = np.array([1.0, np.nan, 3.0, 40.0, 5.0, 6.0, np.nan, 8.0] * 30, dtype=float)
    for program_id in PROGRAM_IDS:
        if program_id == "h_ref":
            continue
        prepared = PoolProgram(program_id).prepare(
            MethodSeriesView(series_uid="u", degraded_inner_train=values),
            None,
            {"period": 24.0},
        )
        assert prepared.units == "original_units"
        assert len(prepared.values) == len(values)
        if program_id != "raw":
            assert np.isfinite(prepared.values).all()


def test_h_ref_has_no_standalone_method_form():
    # H_ref's action depends on the whole role's episode state, not on one series' values.
    # Pretending otherwise would let it be executed on a path no other program takes.
    with pytest.raises(ProgramPoolError):
        PoolProgram("h_ref")
    with pytest.raises(ProgramPoolError):
        apply_program("h_ref", np.arange(100, dtype=float), period=24)


def test_mechanism_lookup_covers_every_program():
    for program_id in PROGRAM_IDS:
        assert mechanism_of(program_id)


class _Assignment:
    def __init__(self, uid: str, dataset_id: str, regime: str, role: SplitRole) -> None:
        self.series_uid = uid
        self.dataset_id = dataset_id
        self.regime_tag = regime
        self.role = role


def test_training_pool_never_mixes_two_datasets():
    """The v0.2 training unit, enforced where it is actually used.

    Under the v0.1 role-pooled unit, one closed-form model saw every dataset's windows, so
    a program applied to COVID moved the weights that scored traffic. `_fit_and_score`
    fits one model per dataset; this asserts that the batch each fit sees is confined to a
    single dataset by construction.
    """
    seen_batches: list[set[str]] = []
    dataset_of = {"a1": "ds_a", "a2": "ds_a", "b1": "ds_b"}
    rng = np.random.default_rng(1)
    series = {uid: 10.0 + rng.standard_normal(400) for uid in dataset_of}

    import SelfEvolvingHarnessTS.benchmark.dev_eval as dev_eval

    real_build = dev_eval.build_windows

    def spy(values_by_uid, normalization_by_uid, **kwargs):
        seen_batches.append({dataset_of[uid] for uid in values_by_uid})
        return real_build(values_by_uid, normalization_by_uid, **kwargs)

    dev_eval.build_windows = spy
    try:
        _fit_and_score(
            {"ds_a": ["a1", "a2"], "ds_b": ["b1"]},
            series,
            series,
            {uid: NormalizationState.fit(values) for uid, values in series.items()},
            {},
            {uid: 1.0 for uid in series},
            {uid: values for uid, values in series.items()},
        )
    finally:
        dev_eval.build_windows = real_build

    assert len(seen_batches) == 2
    for batch_datasets in seen_batches:
        assert len(batch_datasets) == 1, "a training pool contained two datasets"


def _support_a_manifest(rows: list[tuple[str, str, str]]):
    """A real SplitManifest holding only Support-A rows: (uid, group_key, exposure)."""
    from SelfEvolvingHarnessTS.benchmark.split import SplitAssignment, SplitManifest

    assignments = tuple(
        SplitAssignment(
            series_uid=uid,
            dataset_id="ds",
            regime_tag="regime",
            overlap_group=group_key,
            exposure_class=exposure,
            length=400,
            group_key=group_key,
            group_hash_value=0.5,
            role=SplitRole.SUPPORT_A,
            forced_by=None,
            chronological_boundaries=None,
        )
        for uid, group_key, exposure in rows
    )
    return SplitManifest(
        benchmark_version=BENCHMARK_VERSION,
        split_salt="test-salt",
        assignments=assignments,
        u_selected_uids=(),
        inner_split={},
        policies={},
        provenance={},
    )


def test_a_validation_holds_only_certified_virgin_series(monkeypatch):
    """Init-136 fed H_ref. It cannot also be the gate that validates an update to H_ref.

    Without this constraint the promotion gate asks the harness to be judged by data it
    already learned from -- a closed loop that would pass any update that merely memorised
    its own history.
    """
    import SelfEvolvingHarnessTS.benchmark.split as split_module

    rows = [(f"v{i}", f"g_virgin_{i}", "certified_virgin") for i in range(20)]
    rows += [(f"e{i}", f"g_exposed_{i}", "probe_consumed") for i in range(5)]
    rows += [(f"l{i}", f"g_legacy_{i}", "uncertain_legacy_exposure") for i in range(3)]

    # Force the hash to send every group to validation; only the exposure rule may override
    # it. Otherwise a passing test would prove nothing but the luck of the draw.
    monkeypatch.setattr(
        split_module, "support_a_partition", lambda *a, **k: "support_a_validation"
    )
    document = build_support_a_subsplit(_support_a_manifest(rows))

    validation = set(document["members"]["support_a_validation"])
    discovery = set(document["members"]["support_a_discovery"])
    exposed = {uid for uid, _, exposure in rows if exposure != "certified_virgin"}

    assert validation & exposed == set(), "an exposed series reached the promotion gate"
    assert exposed <= discovery
    assert document["n_groups_forced_to_discovery_by_exposure"] == 8
    assert document["schema_version"] == "benchmark-support-a-subsplit/2"


def test_subsplit_refuses_to_emit_an_empty_validation_half(monkeypatch):
    import SelfEvolvingHarnessTS.benchmark.split as split_module

    rows = [(f"e{i}", f"g{i}", "probe_consumed") for i in range(4)]
    monkeypatch.setattr(
        split_module, "support_a_partition", lambda *a, **k: "support_a_validation"
    )
    # Every group is exposed, so every group is forced to discovery and the gate half is
    # empty. That is a roster the protocol cannot validate on, and it must say so loudly.
    with pytest.raises(SplitManifestError, match="came out empty"):
        build_support_a_subsplit(_support_a_manifest(rows))
