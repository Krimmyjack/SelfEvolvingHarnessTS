from SelfEvolvingHarnessTS.methods.h_ref_v02.config import (
    DET_PROGRAM_STEPS,
    H0_ALLOCATION,
    H0_EXPECTED_TOTAL_K,
    HRefState,
    default_state,
)
from SelfEvolvingHarnessTS.p6.harness_state import P6HarnessState


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
