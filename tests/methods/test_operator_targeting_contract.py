import numpy as np

from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.methods.ttha.retrieval import EffectiveHarnessView
from SelfEvolvingHarnessTS.operators.registry import (
    OPERATOR_METADATA,
    OPERATOR_NAMES,
    OPERATOR_TARGETING_MODES,
    operator_targeting_mode,
)
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline
from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate
from SelfEvolvingHarnessTS.runtime.public_features import extract_public_features


def _view() -> EffectiveHarnessView:
    return EffectiveHarnessView(
        instruction="test",
        skills=(),
        memories=(),
        controls={
            "verification": {
                "max_modified_fraction": 0.35,
                "preserve_outside_candidate_region": True,
            }
        },
        effective_harness_view_sha="0" * 64,
    )


def _candidate(operator_id: str, params: dict[str, object]) -> Candidate:
    return Candidate.program_candidate(
        f"candidate-{operator_id}",
        Program.from_steps([(operator_id, params)], source="targeting-contract-test"),
        source="targeting-contract-test",
    )


def _risk_allows(
    candidate: Candidate,
    values: np.ndarray,
    view: EffectiveHarnessView,
    inspected_regions: tuple[tuple[int, int], ...],
) -> bool:
    verification = view.controls["verification"]
    return verify_candidate(
        candidate,
        values,
        allowed_operators=OPERATOR_NAMES,
        inspected_regions=inspected_regions,
        maximum_modified_fraction=float(verification["max_modified_fraction"]),
        preserve_outside_inspected_region=bool(
            verification["preserve_outside_candidate_region"]
        ),
        require_finite_output=False,
    ).selectable


def test_every_canonical_operator_declares_a_valid_targeting_mode():
    for operator_id in OPERATOR_NAMES:
        assert operator_targeting_mode(operator_id) in OPERATOR_TARGETING_MODES


def test_intrinsic_targeter_uses_its_own_hits_instead_of_external_interval():
    values = np.sin(2.0 * np.pi * np.arange(192, dtype=float) / 24.0)
    values[20] += 10.0
    values[120] -= 10.0
    candidate = _candidate("hampel_filter", {"window": 7, "n_sigmas": 3.0})

    assert _risk_allows(candidate, values, _view(), ((112, 128),))


def test_hampel_public_global_gate_only_changes_public_robust_z_hits():
    values = np.sin(2.0 * np.pi * np.arange(192, dtype=float) / 24.0)
    values[20] += 10.0
    values[120] -= 10.0
    result = run_pipeline(
        [
            (
                "hampel_filter",
                {"window": 7, "n_sigmas": 3.0, "global_z_min": 4.0},
            )
        ],
        values,
        source="public-gate-test",
    )
    assert result.ok and result.artifact is not None
    modified = set(np.flatnonzero(result.artifact != values))
    public_hits = set(extract_public_features(values).outlier_indices)
    assert modified
    assert modified <= public_hits


def test_no_canonical_operator_declares_external_region_targeting():
    """Frozen design §7.3/§17.4: RUNTIME_BOUND and OPERATOR_INTRINSIC exclude
    each other, and the one operator that declared ``external_region`` had a
    contract its implementation contradicted.  It now declares the intrinsic
    path it always had, so no operator asks the Runtime to localize for it and
    no Runtime-side localizer needs to exist.
    """
    for operator_id in OPERATOR_NAMES:
        assert operator_targeting_mode(operator_id) != "external_region"
        metadata = OPERATOR_METADATA[operator_id]
        assert not metadata.get("public_parameter_bindings")


def test_intrinsic_level_repair_edits_only_the_excursion_it_finds_itself():
    """The mechanical content of INTRINSIC_GEOMETRY_LOCALIZED.

    Given a window with one transient excursion, the operator with no
    parameters locates that excursion inside the unit it was handed.  The
    legacy call, handed a foreign coordinate system's full-prefix region,
    rewrites most of the same window instead.
    """
    values = np.zeros(192, dtype=float)
    values[40:60] = 3.0

    intrinsic = run_pipeline(
        [("repair_level_shift", {})], values, source="intrinsic-targeting-test"
    )
    assert intrinsic.ok and intrinsic.artifact is not None
    intrinsic_modified = set(np.flatnonzero(intrinsic.artifact != values))

    legacy = run_pipeline(
        [
            (
                "repair_level_shift",
                {
                    "region_start_fraction": 0.001,
                    "region_end_fraction": 0.997,
                    "estimated_offset": 3.0,
                },
            )
        ],
        values,
        source="legacy-targeting-test",
    )
    assert legacy.ok and legacy.artifact is not None
    legacy_modified = set(np.flatnonzero(legacy.artifact != values))

    assert intrinsic_modified
    assert intrinsic_modified <= set(range(40, 60))
    assert len(intrinsic_modified) < len(legacy_modified)


def test_global_transform_still_obeys_external_scope_guard():
    values = np.sin(2.0 * np.pi * np.arange(192, dtype=float) / 24.0)
    values[20] += 5.0
    candidate = _candidate("denoise_median", {"window": 5})

    assert not _risk_allows(candidate, values, _view(), ((16, 24),))
