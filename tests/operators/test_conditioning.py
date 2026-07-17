from SelfEvolvingHarnessTS.conditioning import binning, key as key_module, thresholds
from SelfEvolvingHarnessTS.conditioning.distance import ALPHA_DISTANCE, distance


def test_conditioning_thresholds_are_locally_owned_and_functional():
    assert binning.TH is thresholds
    assert key_module.TH is thresholds
    assert ALPHA_DISTANCE == thresholds.ALPHA_DISTANCE
    assert binning.pattern_bin({"SNR": 3.0, "missing_rate": 0.1}) == "snrLow|miss"
    key = {
        "pattern": {
            "struct_feats": {},
            "quality_profile": {"problem_types": {}, "urgency": 0.0},
        }
    }
    assert distance(key, key) == 0.0
