import numpy as np

from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import _risk_allows
from SelfEvolvingHarnessTS.methods.ttha.retrieval import EffectiveHarnessView
from SelfEvolvingHarnessTS.operators.registry import (
    OPERATOR_NAMES,
    OPERATOR_TARGETING_MODES,
    operator_targeting_mode,
)
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline
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


def test_external_region_targeter_must_remain_inside_inspected_interval():
    values = np.zeros(192, dtype=float)
    values[40:60] = 3.0
    candidate = _candidate(
        "repair_level_shift",
        {
            "region_start_fraction": 40 / 192,
            "region_end_fraction": 60 / 192,
            "estimated_offset": 3.0,
        },
    )

    assert not _risk_allows(candidate, values, _view(), ((100, 120),))


def test_global_transform_still_obeys_external_scope_guard():
    values = np.sin(2.0 * np.pi * np.arange(192, dtype=float) / 24.0)
    values[20] += 5.0
    candidate = _candidate("denoise_median", {"window": 5})

    assert not _risk_allows(candidate, values, _view(), ((16, 24),))
