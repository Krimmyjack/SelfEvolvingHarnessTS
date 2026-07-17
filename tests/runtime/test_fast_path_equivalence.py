import hashlib

import numpy as np

from SelfEvolvingHarnessTS.methods.h_ref_v02.config import (
    DET_PROGRAM_STEPS,
    H0_ALLOCATION,
    H0_EXPECTED_TOTAL_K,
    HRefState,
    default_state,
)
from SelfEvolvingHarnessTS.p6.harness_state import P6HarnessState
from SelfEvolvingHarnessTS.runtime.fast_path import det_ladder, prepared_artifact, run_fast_path


def _views():
    views = {}
    for i in range(3):
        t = np.arange(160, dtype=float)
        rng = np.random.default_rng(100 + i)
        values = np.sin(2.0 * np.pi * t / 24.0 + 0.7 * i) + 0.3 * i
        values = values + rng.normal(0.0, 0.05, t.size)
        if i == 1:
            values[10:14] = np.nan
        views[f"u{i}"] = values
    return views


def _sha(values):
    payload = np.ascontiguousarray(values, dtype="<f8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_h_ref_state_identity_and_frozen_fingerprint():
    state = default_state()
    assert P6HarnessState is HRefState
    assert state.sha() == "4e7e4ac5b40c941d"
    assert state.sampler.allocation == {"det": 3, "random": 5, "llm": 0} == H0_ALLOCATION
    assert state.sampler.expected_total == 8 == H0_EXPECTED_TOTAL_K
    assert DET_PROGRAM_STEPS == (
        (("impute_linear", {}),),
        (("impute_linear", {}), ("winsorize", {}), ("denoise_savgol", {})),
        (("impute_linear", {}), ("denoise_median", {"window": 9})),
    )


def test_fast_path_matches_pre_refactor_fingerprints():
    assert [candidate.sha for candidate in det_ladder()] == [
        "a6a6db644a7b61c0",
        "c0f66a51e987f8a7",
        "bee33065e1b25757",
    ]
    views = _views()
    state = default_state()
    result = run_fast_path(views, state, state.sampler.expected_total)
    expected = {
        "u0": "395193575038668d833b9cbba32b1f2a6ba486cac492ac19e226c2498da41c00",
        "u1": "3734fc053b74086f7d49b18d13ce6b7f0452b1fc24980467870afb1ff2816b19",
        "u2": "23f7c814d104764da16d8a0c62649bf0f97e8dbed95f2b36d4d2997a005dde15",
    }
    for uid, choice in result.items():
        assert choice.sha == "a6a6db644a7b61c0"
        assert result.pool_stats[uid]["realized_pool_size"] == 8
        assert _sha(prepared_artifact(choice, views[uid])) == expected[uid]
